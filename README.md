# Paraformer API 部署指南

## 服务器要求
- 内存：4GB+
- Python：3.10+
- 公网 IP + 开放 8000 端口

---

## 方式一：Docker / Coolify 部署（推荐）

### 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `API_TOKEN` | ✅ | API 鉴权 Token |
| `MODELSCOPE_CACHE` | 可选 | 模型缓存路径，默认 `/data/models` |

### Docker Compose

```bash
# 设置 Token
export API_TOKEN=你的密钥

# 启动（首次会下载模型约 1GB，请耐心等待）
docker compose up -d
```

### Coolify 部署

1. 在 Coolify 中创建新服务，选择 **Docker Compose** 类型
2. 关联此 GitHub 仓库
3. 设置环境变量 `API_TOKEN`
4. **关键：配置持久卷** — 将 `model-cache` 卷挂载到 `/data/models`
   - 这样模型只需首次下载（~1GB），后续重新部署直接使用缓存
5. 部署即可

> ⚠️ 首次启动需要下载约 1GB 的模型文件（paraformer-zh + cam++），请确保部署超时时间足够长。后续重建容器会直接命中卷缓存，秒级启动。

---

## 方式二：裸机部署

```bash
# 1. 安装依赖
pip install fastapi uvicorn funasr modelscope torch torchaudio requests

# 2. 上传 paraformer_api.py 到服务器

# 3. 启动服务（设置你的 Token）
API_TOKEN=你的密钥 python paraformer_api.py
```

首次启动会下载模型（约 1GB），请耐心等待。

### 后台运行

```bash
API_TOKEN=你的密钥 nohup python paraformer_api.py > api.log 2>&1 &
```

---

## 验证

```bash
curl http://服务器IP:8000/health
```

## API 端点

- `GET /health` — 健康检查
- `POST /transcribe` — 上传音频转写（支持 `?diarize=true` 说话人分离）
- `POST /transcribe/url` — URL 音频转写

所有 POST 端点需要 `Authorization: Bearer <token>` 头。
