FROM python:3.10-slim

WORKDIR /app

# 缓存路径
ENV WHISPER_CACHE=/data/models
ENV HF_HOME=/data/models

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git curl libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# 先装 PyTorch (ARM 兼容，CPU 版本)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY whisper_api.py .

EXPOSE 8000

CMD ["python", "whisper_api.py"]
