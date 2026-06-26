"""
CLI test script — Test PaddleOCR trực tiếp trong Docker container (không cần FastAPI server).

Usage trong container:
    python cli_test.py                    # Mặc định test với test/1.jpg
    python cli_test.py test/2.jpg         # Chỉ định file cụ thể
    python cli_test.py /path/to/image.jpg # Đường dẫn tùy ý
"""
import sys
import time
import numpy as np
import paddle
from PIL import Image
from paddleocr import PaddleOCR


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test/1.jpg"

    # --- GPU Detection ---
    use_gpu = False
    try:
        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            paddle.set_device("gpu:0")
            use_gpu = True
            print(f"GPU enabled: {paddle.device.cuda.device_count()} GPU(s) detected → using {paddle.get_device()}")
        else:
            paddle.set_device("cpu")
            print("No GPU detected. Using CPU.")
    except Exception as e:
        paddle.set_device("cpu")
        print(f"GPU detection failed: {e}. Using CPU.")

    # --- 1. Load model ---
    device = "GPU" if use_gpu else "CPU"
    print("=" * 60)
    print(f"Loading PaddleOCR PP-OCRv6 medium model on {device}...")
    t0 = time.time()

    ocr = PaddleOCR(
        use_textline_orientation=True,
        lang="en",
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name="PP-OCRv6_medium_rec",
        use_gpu=use_gpu,
    )

    t_load = time.time()
    print(f"Model loaded in {(t_load - t0) * 1000:.0f}ms")

    # --- 2. Warmup inference (dummy image) ---
    print("Running warmup inference...")
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    try:
        ocr.predict(dummy)
    except Exception:
        pass  # Ảnh trắng có thể không tìm thấy text — OK
    t_warmup = time.time()
    print(f"Warmup done in {(t_warmup - t_load) * 1000:.0f}ms")

    # --- 3. Real inference ---
    print(f"\nProcessing: {image_path}")
    img = np.array(Image.open(image_path).convert("RGB"))
    w, h = img.shape[1], img.shape[0]
    print(f"Image size: {w}x{h}")
    print("-" * 60)

    t_start = time.time()
    result = ocr.predict(img)
    t_end = time.time()

    inference_ms = (t_end - t_start) * 1000

    # --- 4. Print results ---
    texts = []
    if result:
        res = result[0]
        rec_texts = res.get("rec_texts", [])
        rec_scores = res.get("rec_scores", [])

        for text, score in zip(rec_texts, rec_scores):
            texts.append(text)
            print(f"  [{score:.3f}] {text}")

    print("-" * 60)
    print(f"Total texts found : {len(texts)}")
    print(f"Inference time    : {inference_ms:.0f}ms")
    print("=" * 60)

    # --- 5. Device info ---
    print(f"Paddle device     : {paddle.get_device()}")
    print(f"CUDA available    : {paddle.is_compiled_with_cuda()}")


if __name__ == "__main__":
    main()
