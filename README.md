# Paraformer ASR API

语音转写 API 服务，基于阿里 FunASR Paraformer 模型。

## 功能

- **自动预处理**：任意音频格式自动转为 16kHz 单声道 WAV
- **VAD 分段**：内置语音活动检测，长音频自动分段转写
- **标点恢复**：自动添加中文标点符号
- **说话人分离**：可选的多说话人识别（CAM++ 模型）
- **多格式支持**：mp3, wav, m4a, mp4, flac, ogg, webm, wma, aac

## 部署

### 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `API_TOKEN` | ✅ | API 鉴权 Token |
| `MODELSCOPE_CACHE` | ❌ | 模型缓存路径（默认 `/data/models`） |
| `PORT` | ❌ | 服务端口（默认 `8000`） |

### Docker Compose

```bash
API_TOKEN=your_secret_token docker compose up -d --build
```

### Coolify

1. 连接 GitHub 仓库
2. 设置环境变量 `API_TOKEN`
3. 添加持久卷挂载 `/data/models`（模型缓存，约 2GB）
4. 部署

## API

### `GET /health`
健康检查

### `POST /transcribe`
转写上传的音频文件

```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@audio.mp3"

# 启用说话人分离
curl -X POST "http://localhost:8000/transcribe?diarize=true" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@audio.mp3"
```

### `POST /transcribe/url`
从 URL 转写音频（支持 Google Drive）

```bash
curl -X POST "http://localhost:8000/transcribe/url?audio_url=https://example.com/audio.mp3" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## 注意事项

- 首次启动需下载模型（约 1-2GB），请耐心等待
- CPU 推理较慢，9 分钟音频约需 2-5 分钟处理
- 建议挂载持久卷避免重复下载模型
