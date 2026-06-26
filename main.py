import io
import os
import time
import logging
import numpy as np
import paddle
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR, TextDetection

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# --- GPU Detection & Setup ---
USE_GPU = False
try:
    if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
        paddle.set_device("gpu:0")
        USE_GPU = True
        logger.info(f"GPU enabled: {paddle.device.cuda.device_count()} GPU(s) detected. Using device: {paddle.get_device()}")
    else:
        paddle.set_device("cpu")
        logger.info("No GPU detected. Falling back to CPU.")
except Exception as e:
    paddle.set_device("cpu")
    logger.warning(f"GPU detection failed: {e}. Falling back to CPU.")

app = FastAPI(
    title="PaddleOCR PP-OCRv6 API",
    description="FastAPI service for running PaddleOCR locally (PP-OCRv6) with speedups, multiple model tiers, and table formatting",
    version="1.2"
)

# Enable CORS for convenience
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cached instances (only using medium tier)
ocr_instance = None
text_det_instance = None

def get_ocr_instance():
    """
    Lazily loads and caches the PaddleOCR medium instance.
    Automatically uses GPU if available, otherwise CPU.
    """
    global ocr_instance
    if ocr_instance is None:
        device = "GPU" if USE_GPU else "CPU"
        logger.info(f"Initializing PaddleOCR PP-OCRv6 medium engine on {device}...")
        try:
            ocr_instance = PaddleOCR(
                use_textline_orientation=True,
                lang='en',
                text_detection_model_name="PP-OCRv6_medium_det",
                text_recognition_model_name="PP-OCRv6_medium_rec",
            )
            logger.info(f"PaddleOCR medium engine initialized successfully on {device}.")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR on {device}: {e}")
            raise
    return ocr_instance

def get_text_det_instance():
    """
    Lazily loads and caches the PaddleOCR TextDetection medium instance.
    Automatically uses GPU if available, otherwise CPU.
    """
    global text_det_instance
    if text_det_instance is None:
        device = "GPU" if USE_GPU else "CPU"
        logger.info(f"Initializing TextDetection PP-OCRv6 medium engine on {device}...")
        try:
            text_det_instance = TextDetection(model_name="PP-OCRv6_medium_det")
            logger.info(f"TextDetection medium engine initialized successfully on {device}.")
        except Exception as e:
            logger.error(f"Failed to initialize TextDetection on {device}: {e}")
            raise
    return text_det_instance

# Eagerly load the models and run warmup inference during startup
@app.on_event("startup")
def load_default_models():
    device = "GPU" if USE_GPU else "CPU"
    logger.info(f"Starting model loading on {device}...")

    try:
        ocr = get_ocr_instance()
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load OCR model: {e}")
        return

    try:
        det = get_text_det_instance()
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load TextDetection model: {e}")
        return

    # Warmup: chạy inference giả để khởi tạo computation graph + CUDA kernels
    # Giúp loại bỏ cold start hoàn toàn cho request đầu tiên
    logger.info("Running warmup inference to eliminate cold start...")
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    try:
        ocr.predict(dummy_img)
        logger.info("OCR warmup complete.")
    except Exception as e:
        logger.warning(f"OCR warmup error (may be expected for blank image): {e}")

    try:
        list(det.predict(input=dummy_img, batch_size=1))
        logger.info("TextDetection warmup complete.")
    except Exception as e:
        logger.warning(f"TextDetection warmup error (may be expected for blank image): {e}")

    logger.info(f"Paddle device  : {paddle.get_device()}")
    logger.info(f"CUDA available : {paddle.is_compiled_with_cuda()}")
    logger.info(f"All models loaded and warmed up on {device}. Server ready!")

def ocr_to_rows(ocr_results, y_threshold=20):
    """
    Groups OCR bounding boxes into horizontal rows based on vertical proximity.
    Sorts each row from left to right.
    """
    items = []
    for item in ocr_results:
        box = item["box"]
        text = item["text"]
        center_y = sum(pt[1] for pt in box) / 4.0
        min_x = min(pt[0] for pt in box)
        items.append({
            "text": text,
            "center_y": center_y,
            "min_x": min_x
        })
    
    # Sort vertically by center_y
    items.sort(key=lambda x: x["center_y"])
    
    rows = []
    current_row = []
    
    for item in items:
        if not current_row:
            current_row.append(item)
        else:
            # Check vertical alignment against the current row's average center_y
            avg_y = sum(x["center_y"] for x in current_row) / len(current_row)
            if abs(item["center_y"] - avg_y) < y_threshold:
                current_row.append(item)
            else:
                # Complete the current row (sort horizontally by min_x)
                current_row.sort(key=lambda x: x["min_x"])
                rows.append([x["text"] for x in current_row])
                current_row = [item]
                
    if current_row:
        current_row.sort(key=lambda x: x["min_x"])
        rows.append([x["text"] for x in current_row])
        
    return rows

@app.post("/ocr")
async def perform_ocr(
    file: UploadFile = File(None),
    test_file: str = Query(None, description="Select a local test file (e.g. '1.jpg', '2.jpg', '3.jpg', '4.jpg') to test without uploading."),
    max_dim: int = Query(1024, description="Maximum image dimension (width or height) to resize for faster OCR. Set to 0 to keep original size."),
    format: str = Query("raw", pattern="^(raw|table)$", description="Response format. 'raw' for detailed bounding boxes, 'table' for reconstructed table rows."),
    y_threshold: int = Query(20, description="Vertical pixel threshold for grouping text into the same row (only used if format='table').")
):
    t_start = time.time()

    if test_file is None and file is None:
        raise HTTPException(status_code=400, detail="Either 'file' must be uploaded or 'test_file' parameter must be provided.")
    
    try:
        if test_file is not None:
            # Resolve local test file path (works in both local dev and Docker)
            test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test")
            local_path = os.path.join(test_dir, test_file)
            
            # Prevent directory traversal
            if not os.path.abspath(local_path).startswith(os.path.abspath(test_dir)):
                raise HTTPException(status_code=400, detail="Invalid test file path.")
                
            if not os.path.exists(local_path):
                raise HTTPException(status_code=404, detail=f"Local test file '{test_file}' not found.")
                
            image = Image.open(local_path).convert("RGB")
            filename = test_file
        else:
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            filename = file.filename
            
        w_orig, h_orig = image.size
        
        # 1. Pipeline Optimization: Resize image to speed up detection + recognition
        scale_factor = 1.0
        if max_dim > 0 and max(w_orig, h_orig) > max_dim:
            scale_factor = max(w_orig, h_orig) / max_dim
            new_w = int(w_orig / scale_factor)
            new_h = int(h_orig / scale_factor)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        t_preprocess = time.time()
        img_np = np.array(image)
        
        # Resolve the OCR instance
        ocr = get_ocr_instance()
        
        # Run PaddleOCR inference
        ocr_result = ocr.predict(img_np)
        t_inference = time.time()
        
        # Format the response
        results = []
        if ocr_result:
            res_dict = ocr_result[0]
            texts = res_dict.get("rec_texts", [])
            scores = res_dict.get("rec_scores", [])
            polys = res_dict.get("rec_polys", [])
            
            for text, score, poly in zip(texts, scores, polys):
                # Scale box coordinates back to the original image dimensions
                box_cleaned = [[float(coord) * scale_factor for coord in pt] for pt in poly]
                
                results.append({
                    "text": text,
                    "confidence": float(score),
                    "box": box_cleaned
                })
        
        t_total = time.time()
        processing_time = {
            "preprocess_ms": round((t_preprocess - t_start) * 1000, 2),
            "inference_ms": round((t_inference - t_preprocess) * 1000, 2),
            "postprocess_ms": round((t_total - t_inference) * 1000, 2),
            "total_ms": round((t_total - t_start) * 1000, 2),
        }
        
        # Log to server terminal
        logger.info(f"[/ocr] file={filename} | size={w_orig}x{h_orig} | processed={image.size[0]}x{image.size[1]} | texts={len(results)} | {processing_time}")
        
        # 2. Output Formatting
        if format == "table":
            rows = ocr_to_rows(results, y_threshold=y_threshold)
            return {
                "status": "success",
                "filename": filename,
                "tier": "medium",
                "format": "table",
                "dimensions": {"original": [w_orig, h_orig], "processed": list(image.size)},
                "processing_time_ms": processing_time,
                "rows": rows
            }
        else:
            return {
                "status": "success",
                "filename": filename,
                "tier": "medium",
                "format": "raw",
                "dimensions": {"original": [w_orig, h_orig], "processed": list(image.size)},
                "processing_time_ms": processing_time,
                "results": results
            }
            
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

@app.post("/detect")
async def perform_detection(
    file: UploadFile = File(None),
    test_file: str = Query(None, description="Select a local test file (e.g. '1.jpg', '2.jpg', '3.jpg', '4.jpg') to test without uploading."),
    max_dim: int = Query(1024, description="Maximum image dimension (width or height) to resize for faster OCR. Set to 0 to keep original size.")
):
    t_start = time.time()
    
    if test_file is None and file is None:
        raise HTTPException(status_code=400, detail="Either 'file' must be uploaded or 'test_file' parameter must be provided.")
    
    try:
        if test_file is not None:
            # Resolve local test file path (works in both local dev and Docker)
            test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test")
            local_path = os.path.join(test_dir, test_file)
            
            # Prevent directory traversal
            if not os.path.abspath(local_path).startswith(os.path.abspath(test_dir)):
                raise HTTPException(status_code=400, detail="Invalid test file path.")
                
            if not os.path.exists(local_path):
                raise HTTPException(status_code=404, detail=f"Local test file '{test_file}' not found.")
                
            image = Image.open(local_path).convert("RGB")
            filename = test_file
        else:
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            filename = file.filename
            
        w_orig, h_orig = image.size
        
        # Resize image if it exceeds max_dim to speed up detection
        scale_factor = 1.0
        if max_dim > 0 and max(w_orig, h_orig) > max_dim:
            scale_factor = max(w_orig, h_orig) / max_dim
            new_w = int(w_orig / scale_factor)
            new_h = int(h_orig / scale_factor)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        t_preprocess = time.time()
        img_np = np.array(image)
        
        # Load the text detection model
        model = get_text_det_instance()
        output = model.predict(input=img_np, batch_size=1)
        t_inference = time.time()
        
        results = []
        for res in output:
            polys = res.get("dt_polys", [])
            scores = res.get("dt_scores", [])
            
            # Convert numpy types to native types and scale back coordinates
            for poly, score in zip(polys, scores):
                box_cleaned = [[float(coord) * scale_factor for coord in pt] for pt in poly]
                results.append({
                    "confidence": float(score),
                    "box": box_cleaned
                })
        
        t_total = time.time()
        processing_time = {
            "preprocess_ms": round((t_preprocess - t_start) * 1000, 2),
            "inference_ms": round((t_inference - t_preprocess) * 1000, 2),
            "postprocess_ms": round((t_total - t_inference) * 1000, 2),
            "total_ms": round((t_total - t_start) * 1000, 2),
        }
        
        # Log to server terminal
        logger.info(f"[/detect] file={filename} | size={w_orig}x{h_orig} | processed={image.size[0]}x{image.size[1]} | regions={len(results)} | {processing_time}")
                
        return {
            "status": "success",
            "filename": filename,
            "tier": "medium",
            "dimensions": {"original": [w_orig, h_orig], "processed": list(image.size)},
            "processing_time_ms": processing_time,
            "results": results
        }
            
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text detection failed: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}
