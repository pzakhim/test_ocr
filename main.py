import io
import os
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR, TextDetection

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
    """
    global ocr_instance
    if ocr_instance is None:
        print("Initializing PaddleOCR PP-OCRv6 medium engine...")
        ocr_instance = PaddleOCR(
            use_textline_orientation=True,
            lang='en',
            text_detection_model_name="PP-OCRv6_medium_det",
            text_recognition_model_name="PP-OCRv6_medium_rec"
        )
        print("PaddleOCR medium engine initialized successfully.")
    return ocr_instance

def get_text_det_instance():
    """
    Lazily loads and caches the PaddleOCR TextDetection medium instance.
    """
    global text_det_instance
    if text_det_instance is None:
        print("Initializing PaddleOCR TextDetection PP-OCRv6 medium engine...")
        text_det_instance = TextDetection(model_name="PP-OCRv6_medium_det")
        print("PaddleOCR TextDetection medium engine initialized successfully.")
    return text_det_instance

# Eagerly load the models during startup
@app.on_event("startup")
def load_default_models():
    get_ocr_instance()
    get_text_det_instance()

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
            
        img_np = np.array(image)
        
        # Resolve the OCR instance
        ocr = get_ocr_instance()
        
        # Run PaddleOCR inference
        ocr_result = ocr.predict(img_np)
        
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
        
        # 2. Output Formatting
        if format == "table":
            rows = ocr_to_rows(results, y_threshold=y_threshold)
            return {
                "status": "success",
                "filename": filename,
                "tier": "medium",
                "format": "table",
                "dimensions": {"original": [w_orig, h_orig], "processed": list(image.size)},
                "rows": rows
            }
        else:
            return {
                "status": "success",
                "filename": filename,
                "tier": "medium",
                "format": "raw",
                "dimensions": {"original": [w_orig, h_orig], "processed": list(image.size)},
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
    import os
    
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
            
        img_np = np.array(image)
        
        # Load the text detection model
        model = get_text_det_instance()
        output = model.predict(input=img_np, batch_size=1)
        
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
                
        return {
            "status": "success",
            "filename": filename,
            "tier": "medium",
            "dimensions": {"original": [w_orig, h_orig], "processed": list(image.size)},
            "results": results
        }
            
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text detection failed: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}
