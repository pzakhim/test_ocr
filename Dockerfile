# Sử dụng Python 3.12-slim để đảm bảo tính ổn định và tương thích của PaddlePaddle
FROM python:3.12-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho PaddleOCR và OpenCV (để vẽ hình)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt công cụ quản lý package uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy thông tin dependency vào trước để tận dụng Docker cache
COPY pyproject.toml uv.lock ./

# Cài đặt dependencies sử dụng Python 3.12
RUN uv sync --frozen --no-cache --python 3.12

# Thêm virtualenv của uv vào biến môi trường PATH
ENV PATH="/app/.venv/bin:$PATH"

# --- BƯỚC QUAN TRỌNG: Tải trước các model weights vào cache lúc build để chạy offline hoàn toàn ---
RUN python -c "from paddleocr import TextDetection, PaddleOCR; \
TextDetection(model_name='PP-OCRv6_medium_det'); \
PaddleOCR(use_textline_orientation=True, lang='en', text_detection_model_name='PP-OCRv6_medium_det', text_recognition_model_name='PP-OCRv6_medium_rec')"

# Copy toàn bộ mã nguồn vào image
COPY . .

# Mở port 8000 cho FastAPI Web Service
EXPOSE 8000

# Khởi chạy FastAPI Web Service mặc định
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
