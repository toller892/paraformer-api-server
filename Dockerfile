FROM python:3.10-slim

WORKDIR /app

# 设置模型缓存路径（可通过环境变量覆盖）
ENV MODELSCOPE_CACHE=/data/models

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY paraformer_api.py .

EXPOSE 8000

CMD ["python", "paraformer_api.py"]
