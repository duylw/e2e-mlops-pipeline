FROM python:3.12-slim

# Copy uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy files phục vụ cho việc cài đặt dependencies
COPY pyproject.toml uv.lock ./

# Xây dựng môi trường ảo bằng uv sync --locked
RUN uv sync --locked

# Trỏ PATH vào môi trường ảo vừa tạo
ENV PATH="/app/.venv/bin:$PATH"

# Chỉ copy mã nguồn phục vụ cho việc serving và preprocessing tương ứng
COPY src/serving ./src/serving
COPY src/preprocessing.py src/utils.py ./src/

# Expose port uvicorn
EXPOSE 8000

# Khởi chạy FastAPI API server
CMD ["uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]