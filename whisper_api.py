#!/usr/bin/env python3
"""
Whisper ASR API 服务 v4.0
- 基于 OpenAI Whisper large-v3-turbo
- 自动音频预处理
- Token 鉴权
- 支持文件上传和 URL 转写
"""
import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

import requests
import uvicorn
import whisper
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ============ 配置 ============
API_TOKEN = os.getenv("API_TOKEN", "")
PORT = int(os.getenv("PORT", 8000))
MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WHISPER_CACHE = os.getenv("WHISPER_CACHE", "/data/models")

# 设置 Whisper 下载目录
os.environ["XDG_CACHE_HOME"] = WHISPER_CACHE
os.makedirs(WHISPER_CACHE, exist_ok=True)

# ============ 模型（异步加载，避免 Coolify 健康检查超时） ============
model = None
model_ready = threading.Event()
model_error = None


def _load_model():
    global model, model_error
    try:
        print(f"正在加载 Whisper {MODEL_NAME} 模型...")
        model = whisper.load_model(MODEL_NAME, download_root=WHISPER_CACHE)
        print(f"✅ Whisper {MODEL_NAME} 加载完成！设备: {model.device}")
    except Exception as e:
        model_error = str(e)
        print(f"❌ 模型加载失败: {e}")
    finally:
        model_ready.set()


threading.Thread(target=_load_model, daemon=True).start()

# ============ FastAPI 应用 ============
app = FastAPI(
    title="Whisper ASR API",
    description="语音转写 API（基于 OpenAI Whisper large-v3-turbo）",
    version="4.0.0",
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


def transcribe_audio(file_path: str, language: str = "zh") -> dict:
    """
    用 Whisper 转写音频。
    Whisper 内部会自动处理采样率转换，无需手动预处理。
    """
    result = model.transcribe(
        file_path,
        language=language,
        verbose=False,
    )

    text = result.get("text", "").strip()

    # 提取分段信息
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        })

    return {
        "text": text,
        "segments": segments,
        "language": result.get("language", language),
    }


# ============ API 端点 ============
@app.get("/")
async def root():
    return {
        "status": "ready" if model_ready.is_set() and model else "loading",
        "service": "Whisper ASR API",
        "version": "4.0.0",
        "model": MODEL_NAME,
        "device": str(model.device) if model else "loading",
    }


@app.get("/health")
async def health():
    if model_error:
        raise HTTPException(503, f"模型加载失败: {model_error}")
    if not model_ready.is_set():
        return {"status": "loading", "model": MODEL_NAME}
    return {"status": "healthy", "model": MODEL_NAME}


def _require_model():
    """确保模型已加载，否则返回 503"""
    if not model_ready.is_set():
        raise HTTPException(503, "模型正在加载中，请稍后重试")
    if model_error or model is None:
        raise HTTPException(503, f"模型不可用: {model_error}")


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Query("zh", description="语言代码，如 zh, en, ja"),
    _: bool = Depends(verify_token),
):
    """
    转写上传的音频文件

    Headers:
        Authorization: Bearer <your_token>

    Query:
        language: 语言代码 (默认 zh)

    Body:
        file: 音频文件 (mp3, wav, m4a, mp4, flac, ogg, webm, wma, aac)
    """
    allowed_ext = {".mp3", ".wav", ".m4a", ".mp4", ".flac", ".ogg", ".webm", ".wma", ".aac"}
    file_ext = Path(file.filename).suffix.lower() if file.filename else ".mp3"

    if file_ext not in allowed_ext:
        raise HTTPException(400, f"不支持的格式: {file_ext}，支持: {', '.join(sorted(allowed_ext))}")

    tmp_path = None
    try:
        _require_model()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        result = transcribe_audio(tmp_path, language=language)

        if not result or not result.get("text"):
            raise HTTPException(500, "转写失败：未识别到语音内容")

        return JSONResponse({
            "success": True,
            "text": result["text"],
            "segments": result["segments"],
            "language": result["language"],
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"处理错误: {str(e)}")
    finally:
        cleanup_files(tmp_path)


@app.post("/transcribe/url")
async def transcribe_url(
    audio_url: str,
    language: str = Query("zh", description="语言代码，如 zh, en, ja"),
    _: bool = Depends(verify_token),
):
    """
    从 URL 转写音频（支持 Google Drive）

    Headers:
        Authorization: Bearer <your_token>

    Query:
        language: 语言代码 (默认 zh)

    Body:
        audio_url: 音频文件 URL
    """
    tmp_path = None
    try:
        _require_model()
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

        result = transcribe_audio(tmp_path, language=language)

        if not result or not result.get("text"):
            raise HTTPException(500, "转写失败：未识别到语音内容")

        return JSONResponse({
            "success": True,
            "text": result["text"],
            "segments": result["segments"],
            "language": result["language"],
        })

    except requests.RequestException as e:
        raise HTTPException(400, f"下载失败: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"处理错误: {str(e)}")
    finally:
        cleanup_files(tmp_path)


if __name__ == "__main__":
    if not API_TOKEN:
        print("⚠️  警告: 未设置 API_TOKEN 环境变量！")
        print("   请使用: API_TOKEN=your_secret_token python whisper_api.py")
        exit(1)

    print(f"🚀 Whisper ASR API 启动在 http://0.0.0.0:{PORT}")
    print(f"   模型: {MODEL_NAME} (异步加载中...)")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
