# Paraformer API 部署指南

## 服务器要求
- 内存：4GB+
- Python：3.10+
- 公网 IP + 开放 8000 端口

## 部署步骤

```bash
# 1. 安装依赖
pip install fastapi uvicorn funasr modelscope torch torchaudio requests

# 2. 上传 paraformer_api.py 到服务器

# 3. 启动服务（设置你的 Token）
API_TOKEN=你的密钥 python paraformer_api.py
```

首次启动会下载模型（约 1GB），请耐心等待。

## 后台运行

```bash
API_TOKEN=你的密钥 nohup python paraformer_api.py > api.log 2>&1 &
```

## 验证

```bash
curl http://服务器IP:8000/health
```

## 完成后提供
- 服务器公网 IP
- 端口：8000
- Token：你设置的密钥
