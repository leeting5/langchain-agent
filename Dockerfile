FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# 启动 REST API 服务（src/api/app.py）
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]