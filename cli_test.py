import os
import sys
import time
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

# Disable MKLDNN (oneDNN) to prevent PIR compatibility errors on CPU
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

def run_test(image_path: str):
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        sys.exit(1)
        
    print(f"Loading image: {image_path}...")
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
        
    w, h = image.size
    print(f"Image dimensions: {w}x{h}")
    
    # Optional resizing to match main.py behavior (e.g. max_dim=1024)
    max_dim = 1024
    if max(w, h) > max_dim:
        scale = max(w, h) / max_dim
        new_w, new_h = int(w / scale), int(h / scale)
        print(f"Resizing from {w}x{h} to {new_w}x{new_h} for optimization...")
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
    img_np = np.array(image)
    
    print("\n--- Initializing PaddleOCR PP-OCRv6 medium ---")
    t0 = time.time()
    # This will load from cache (downloaded during docker build or local init)
    model = PaddleOCR(
        use_textline_orientation=True,
        lang='en',
        text_detection_model_name='PP-OCRv6_medium_det',
        text_recognition_model_name='PP-OCRv6_medium_rec'
    )
    t_init = time.time()
    print(f"Initialization took: {round((t_init - t0) * 1000, 2)} ms")
    
    print("\n--- Running Inference ---")
    t1 = time.time()
    # Correct way to predict in PP-OCRv6 to avoid paddlex errors
    result = model.predict(img_np)
    t2 = time.time()
    
    print(f"Inference took: {round((t2 - t1) * 1000, 2)} ms (~{round(t2 - t1, 2)} seconds)")
    
    print("\n--- Results ---")
    if result:
        res_dict = result[0]
        texts = res_dict.get("rec_texts", [])
        scores = res_dict.get("rec_scores", [])
        polys = res_dict.get("rec_polys", [])
        
        for idx, (text, score) in enumerate(zip(texts, scores)):
            print(f"[{idx+1}] Text: {text} | Confidence: {round(score, 4)}")
    else:
        print("No text detected.")

if __name__ == "__main__":
    # Default to test/1.jpg if no image path provided
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test/1.jpg"
    run_test(img_path)
