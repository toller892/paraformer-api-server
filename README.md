# Whisper ASR API Server

语音转写 API 服务，基于 OpenAI Whisper large-v3-turbo，支持说话人分离。

## 功能

- ✅ Whisper large-v3-turbo 转写
- ✅ 说话人分离 (pyannote-audio)
- ✅ 多格式支持 (mp3, wav, m4a, mp4, flac, ogg, webm, wma, aac)
- ✅ Token 认证
- ✅ URL 转写（支持 Google Drive）

## 环境变量

| 变量 | 必须 | 说明 |
|------|------|------|
| `API_TOKEN` | ✅ | API 认证 Token |
| `HF_TOKEN` | ❌ | HuggingFace Token（启用说话人分离需要） |
| `WHISPER_MODEL` | ❌ | Whisper 模型，默认 `large-v3-turbo` |
| `PORT` | ❌ | 服务端口，默认 `8000` |

### 获取 HuggingFace Token

1. 注册 [HuggingFace](https://huggingface.co/)
2. 同意 [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) 的使用协议
3. 同意 [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) 的使用协议
4. 在 [Settings > Access Tokens](https://huggingface.co/settings/tokens) 创建 Token

## API 端点

### GET /health
健康检查

```json
{
  "status": "healthy",
  "model": "large-v3-turbo",
  "diarization": "available"
}
```

### POST /transcribe
转写上传的音频文件

**Headers:**
- `Authorization: Bearer <your_token>`

**Query:**
- `language`: 语言代码，如 zh, en, ja（默认 zh）
- `diarize`: 是否启用说话人分离（默认 false）

**Body:**
- `file`: 音频文件

**Response:**
```json
{
  "success": true,
  "text": "完整转写文本",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "你好",
      "speaker": "SPEAKER_00"
    }
  ],
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "language": "zh"
}
```

### POST /transcribe/url
从 URL 转写音频（支持 Google Drive）

## Docker 部署

```bash
docker build -t whisper-api .
docker run -d \
  -p 8000:8000 \
  -e API_TOKEN=your_token \
  -e HF_TOKEN=your_hf_token \
  -v whisper-models:/data/models \
  whisper-api
```

## 示例

```bash
# 基础转写
curl -X POST http://localhost:8000/transcribe \
  -H "Authorization: Bearer your_token" \
  -F "file=@audio.mp3" \
  -F "language=zh"

# 带说话人分离
curl -X POST "http://localhost:8000/transcribe?diarize=true" \
  -H "Authorization: Bearer your_token" \
  -F "file=@meeting.mp3" \
  -F "language=zh"
```
