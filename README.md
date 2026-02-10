# Whisper ASR API

基于 OpenAI Whisper large-v3-turbo 的语音转写 API 服务。

## 特性

- 🎙️ Whisper large-v3-turbo 模型（高质量中文转写）
- 🔐 Bearer Token 鉴权
- 📁 支持文件上传和 URL 转写
- 🌐 支持多语言（默认中文）
- 📊 返回分段时间戳
- 🐳 Docker 一键部署

## 快速开始

```bash
# 设置 API Token
export API_TOKEN=your_secret_token

# Docker Compose 启动
docker compose up -d --build

# 查看日志（首次启动需下载 ~1.5GB 模型）
docker compose logs -f
```

## API 接口

### 健康检查

```bash
curl http://localhost:8000/health
```

### 转写音频文件

```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Authorization: Bearer your_secret_token" \
  -F "file=@audio.mp3" \
  -F "language=zh"
```

### 从 URL 转写

```bash
curl -X POST "http://localhost:8000/transcribe/url?audio_url=https://example.com/audio.mp3&language=zh" \
  -H "Authorization: Bearer your_secret_token"
```

## 响应格式

```json
{
  "success": true,
  "text": "完整转写文本",
  "segments": [
    {"start": 0.0, "end": 3.5, "text": "分段文本"},
    ...
  ],
  "language": "zh"
}
```

## 支持的音频格式

mp3, wav, m4a, mp4, flac, ogg, webm, wma, aac

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `API_TOKEN` | (必填) | API 鉴权 Token |
| `WHISPER_MODEL` | `large-v3-turbo` | Whisper 模型名称 |
| `WHISPER_CACHE` | `/data/models` | 模型缓存路径 |
| `PORT` | `8000` | 服务端口 |

## 资源需求

- **模型大小**: ~1.5GB (large-v3-turbo)
- **内存**: ~4-6GB (CPU FP32)
- **CPU**: 推理速度约 2-3x 实时（无 GPU）
- **GPU**: 如有 CUDA GPU，速度可提升 10-20x
