#!/usr/bin/env python3
"""
Paraformer API 服务 v3.0
- 自动预处理：任意音频 → 16kHz 单声道 WAV
- 长音频 VAD 分段转写，避免解码退化
- 支持说话人分离
- Token 鉴权
"""
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import requests
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from funasr import AutoModel

# ============ 配置 ============
API_TOKEN = os.getenv("API_TOKEN", "")

MODELSCOPE_CACHE = os.getenv("MODELSCOPE_CACHE", "")
if MODELSCOPE_CACHE:
    os.environ["MODELSCOPE_CACHE"] = MODELSCOPE_CACHE
    os.makedirs(MODELSCOPE_CACHE, exist_ok=True)

PORT = int(os.getenv("PORT", 8000))

# 单次转写最大时长（秒），超过则分段
MAX_CHUNK_SECONDS = 300

# ============ 初始化模型 ============
print("正在加载 Paraformer 语音识别模型（含 VAD + 标点恢复）...")
asr_model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    disable_update=True,
)
print("✅ Paraformer + VAD + 标点模型加载完成！")

print("正在加载 CAM++ 说话人分离模型...")
spk_model = AutoModel(model="cam++", model_revision="master", disable_update=True)
print("✅ CAM++ 模型加载完成！")

# ============ FastAPI 应用 ============
app = FastAPI(
    title="Paraformer ASR API",
    description="语音转写 API（自动预处理 + VAD 分段 + 标点恢复 + 说话人分离）",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Token 验证 ============
def verify_token(authorization: Optional[str] = Header(None)):
    if not API_TOKEN:
        raise HTTPException(500, "服务器未配置 API_TOKEN")
    if not authorization:
        raise HTTPException(401, "缺少 Authorization 头")
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Authorization 格式错误，需要 Bearer Token")
    if authorization[7:] != API_TOKEN:
        raise HTTPException(403, "Token 无效")
    return True


# ============ 音频预处理 ============
def preprocess_audio(input_path: str) -> str:
    """
    将任意音频转换为 16kHz 单声道 WAV（Paraformer 要求的格式）。
    返回临时 WAV 文件路径（调用方负责清理）。
    """
    output_path = input_path + ".16k.wav"
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000",   # 16kHz 采样率
        "-ac", "1",       # 单声道
        "-sample_fmt", "s16",  # 16-bit PCM
        "-f", "wav",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 预处理失败: {result.stderr[-500:]}")
    return output_path


def get_audio_duration(file_path: str) -> float:
    """获取音频时长（秒）"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def split_audio(file_path: str, chunk_seconds: int = MAX_CHUNK_SECONDS) -> list:
    """
    将长音频按固定时长切分为多个片段。
    返回临时文件路径列表（调用方负责清理）。
    """
    duration = get_audio_duration(file_path)
    if duration <= chunk_seconds:
        return [file_path]

    chunks = []
    start = 0
    idx = 0
    while start < duration:
        chunk_path = f"{file_path}.chunk{idx}.wav"
        cmd = [
            "ffmpeg", "-y", "-i", file_path,
            "-ss", str(start),
            "-t", str(chunk_seconds),
            "-c", "copy",
            chunk_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and os.path.getsize(chunk_path) > 100:
            chunks.append(chunk_path)
        start += chunk_seconds
        idx += 1

    return chunks if chunks else [file_path]


# ============ 工具函数 ============
def convert_gdrive_url(url: str) -> str:
    """将 Google Drive 分享链接转换为直接下载链接"""
    if "export=download" in url:
        return url
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            file_id = match.group(1)
            return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url


def cleanup_files(*paths):
    """安全清理临时文件"""
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.unlink(p)
            except OSError:
                pass


def transcribe_audio(file_path: str, diarize: bool = False) -> dict:
    """
    转写音频（已预处理为 16kHz WAV）。
    模型已集成 VAD + 标点，会自动做语音活动检测和标点恢复。
    长音频额外做分段保险。
    """
    duration = get_audio_duration(file_path)
    chunks = split_audio(file_path)
    temp_chunks = [c for c in chunks if c != file_path]

    try:
        all_text = []
        all_utterances = []

        for chunk_path in chunks:
            if not diarize:
                result = asr_model.generate(input=chunk_path, batch_size_s=300)
                if result and isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict):
                        text = result[0].get("text", "")
                    else:
                        text = str(result[0])
                    if text.strip():
                        all_text.append(text.strip())
            else:
                asr_result = asr_model.generate(
                    input=chunk_path,
                    batch_size_s=300,
                    sentence_timestamp=True,
                )
                spk_result = spk_model.generate(input=chunk_path)

                if asr_result and isinstance(asr_result, list) and len(asr_result) > 0:
                    asr_data = asr_result[0]
                    sentences = asr_data.get("sentence_info", [])
                    if not sentences and "text" in asr_data:
                        text = asr_data["text"]
                        all_text.append(text)
                        all_utterances.append({
                            "speaker": "speaker_0",
                            "start": 0,
                            "end": 0,
                            "text": text,
                        })
                    else:
                        for sent in sentences:
                            spk_id = (
                                f"speaker_{sent.get('spk', 0)}"
                                if "spk" in sent
                                else "speaker_0"
                            )
                            text = sent.get("text", "")
                            all_text.append(text)
                            all_utterances.append({
                                "speaker": spk_id,
                                "start": sent.get("start", 0) / 1000,
                                "end": sent.get("end", 0) / 1000,
                                "text": text,
                            })

        full_text = "".join(all_text)
        response = {"text": full_text}
        if diarize:
            response["utterances"] = all_utterances
        return response

    finally:
        cleanup_files(*temp_chunks)


# ============ API 端点 ============
@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "Paraformer ASR API",
        "version": "3.0.0",
        "features": ["vad", "punctuation", "diarization", "auto-preprocess"],
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    diarize: bool = Query(False, description="是否启用说话人分离"),
    _: bool = Depends(verify_token),
):
    """
    转写上传的音频文件（自动预处理为 16kHz 单声道）

    Headers:
        Authorization: Bearer <your_token>

    Query:
        diarize: 是否启用说话人分离 (默认 false)

    Body:
        file: 音频文件 (mp3, wav, m4a, mp4, flac, ogg, webm, wma, aac)
    """
    allowed_ext = {".mp3", ".wav", ".m4a", ".mp4", ".flac", ".ogg", ".webm", ".wma", ".aac"}
    file_ext = Path(file.filename).suffix.lower() if file.filename else ".mp3"

    if file_ext not in allowed_ext:
        raise HTTPException(400, f"不支持的格式: {file_ext}，支持: {', '.join(sorted(allowed_ext))}")

    tmp_path = None
    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 预处理：转为 16kHz 单声道 WAV
        wav_path = preprocess_audio(tmp_path)

        result = transcribe_audio(wav_path, diarize=diarize)

        if not result or not result.get("text"):
            raise HTTPException(500, "转写失败：未识别到语音内容")

        response = {"success": True, "text": result["text"]}
        if "utterances" in result:
            response["utterances"] = result["utterances"]

        return JSONResponse(response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"处理错误: {str(e)}")
    finally:
        cleanup_files(tmp_path, wav_path)


@app.post("/transcribe/url")
async def transcribe_url(
    audio_url: str,
    diarize: bool = Query(False, description="是否启用说话人分离"),
    _: bool = Depends(verify_token),
):
    """
    从 URL 转写音频（支持 Google Drive，自动预处理）

    Headers:
        Authorization: Bearer <your_token>

    Query:
        diarize: 是否启用说话人分离 (默认 false)

    Body:
        audio_url: 音频文件 URL
    """
    tmp_path = None
    wav_path = None
    try:
        download_url = convert_gdrive_url(audio_url)

        resp = requests.get(download_url, timeout=300, stream=True)
        resp.raise_for_status()

        file_ext = Path(audio_url).suffix.lower()
        if not file_ext or file_ext not in {".mp3", ".wav", ".m4a", ".mp4", ".flac", ".ogg"}:
            file_ext = ".mp3"

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            for chunk in resp.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp_path = tmp.name

        wav_path = preprocess_audio(tmp_path)
        result = transcribe_audio(wav_path, diarize=diarize)

        if not result or not result.get("text"):
            raise HTTPException(500, "转写失败：未识别到语音内容")

        response = {"success": True, "text": result["text"]}
        if "utterances" in result:
            response["utterances"] = result["utterances"]

        return JSONResponse(response)

    except requests.RequestException as e:
        raise HTTPException(400, f"下载失败: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"处理错误: {str(e)}")
    finally:
        cleanup_files(tmp_path, wav_path)


if __name__ == "__main__":
    if not API_TOKEN:
        print("⚠️  警告: 未设置 API_TOKEN 环境变量！")
        print("   请使用: API_TOKEN=your_secret_token python paraformer_api.py")
        exit(1)

    print(f"🚀 服务启动在 http://0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
