FROM python:3.10-slim

WORKDIR /app

# 设置模型缓存路径
ENV MODELSCOPE_CACHE=/data/models

# 安装系统依赖（ffmpeg 用于音频预处理和格式转换）
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY paraformer_api.py .

EXPOSE 8000

CMD ["python", "paraformer_api.py"]
