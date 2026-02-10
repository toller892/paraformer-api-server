FROM python:3.10-slim

WORKDIR /app

# Whisper 模型缓存路径
ENV WHISPER_CACHE=/data/models

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git curl && \
    rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY whisper_api.py .

EXPOSE 8000

CMD ["python", "whisper_api.py"]
