#!/usr/bin/env python3
"""
Whisper ASR API 服务 v5.0
- 基于 OpenAI Whisper large-v3-turbo
- 说话人分离 (pyannote-audio)
- Token 鉴权
- 支持文件上传和 URL 转写
"""
import os
import re
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
HF_TOKEN = os.getenv("HF_TOKEN", "")  # HuggingFace token for pyannote

# 设置缓存目录
os.environ["XDG_CACHE_HOME"] = WHISPER_CACHE
os.environ["HF_HOME"] = WHISPER_CACHE
os.makedirs(WHISPER_CACHE, exist_ok=True)

# ============ 模型（异步加载） ============
whisper_model = None
diarization_pipeline = None
model_ready = threading.Event()
model_error = None


def _load_models():
    global whisper_model, diarization_pipeline, model_error
    try:
        # 加载 Whisper
        print(f"正在加载 Whisper {MODEL_NAME} 模型...")
        whisper_model = whisper.load_model(MODEL_NAME, download_root=WHISPER_CACHE)
        print(f"✅ Whisper {MODEL_NAME} 加载完成！设备: {whisper_model.device}")

        # 加载 pyannote 说话人分离
        if HF_TOKEN:
            print("正在加载 pyannote 说话人分离模型...")
            try:
                from pyannote.audio import Pipeline
                diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=HF_TOKEN,
                    cache_dir=WHISPER_CACHE,
                )
                # CPU 模式
                import torch
                diarization_pipeline.to(torch.device("cpu"))
                print("✅ pyannote 说话人分离模型加载完成！")
            except Exception as e:
                print(f"⚠️ pyannote 加载失败（说话人分离不可用）: {e}")
                diarization_pipeline = None
        else:
            print("⚠️ 未设置 HF_TOKEN，说话人分离功能不可用")

    except Exception as e:
        model_error = str(e)
        print(f"❌ 模型加载失败: {e}")
    finally:
        model_ready.set()


threading.Thread(target=_load_models, daemon=True).start()

# ============ FastAPI 应用 ============
app = FastAPI(
    title="Whisper ASR API",
    description="语音转写 API（Whisper large-v3-turbo + 说话人分离）",
    version="5.0.0",
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


def assign_speakers_to_segments(whisper_segments: list, diarization) -> list:
    """
    将 pyannote 的说话人标签分配给 Whisper 的分段。
    使用重叠时间最长的说话人。
    """
    result = []
    for seg in whisper_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        
        # 找与该分段重叠最多的说话人
        speaker_overlap = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            overlap_start = max(seg_start, turn.start)
            overlap_end = min(seg_end, turn.end)
            if overlap_start < overlap_end:
                overlap = overlap_end - overlap_start
                speaker_overlap[speaker] = speaker_overlap.get(speaker, 0) + overlap
        
        # 选重叠最多的
        if speaker_overlap:
            best_speaker = max(speaker_overlap, key=speaker_overlap.get)
        else:
            best_speaker = "UNKNOWN"
        
        result.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "speaker": best_speaker,
        })
    
    return result


def transcribe_audio(file_path: str, language: str = "zh", diarize: bool = False) -> dict:
    """
    用 Whisper 转写音频，可选说话人分离。
    """
    # Whisper 转写
    result = whisper_model.transcribe(
        file_path,
        language=language,
        verbose=False,
    )

    text = result.get("text", "").strip()

    # 基础分段
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        })

    # 说话人分离
    speakers = []
    if diarize and diarization_pipeline:
        try:
            print("正在进行说话人分离...")
            diarization = diarization_pipeline(file_path)
            segments = assign_speakers_to_segments(segments, diarization)
            # 提取唯一说话人列表
            speakers = sorted(set(s["speaker"] for s in segments if s["speaker"] != "UNKNOWN"))
            print(f"✅ 说话人分离完成，检测到 {len(speakers)} 位说话人")
        except Exception as e:
            print(f"⚠️ 说话人分离失败: {e}")
            # 保留原始分段，不加 speaker

    return {
        "text": text,
        "segments": segments,
        "language": result.get("language", language),
        "speakers": speakers,
    }


# ============ API 端点 ============
@app.get("/")
async def root():
    return {
        "status": "ready" if model_ready.is_set() and whisper_model else "loading",
        "service": "Whisper ASR API",
        "version": "5.0.0",
        "model": MODEL_NAME,
        "device": str(whisper_model.device) if whisper_model else "loading",
        "diarization": "available" if diarization_pipeline else "unavailable",
    }


@app.get("/health")
async def health():
    if model_error:
        raise HTTPException(503, f"模型加载失败: {model_error}")
    if not model_ready.is_set():
        return {"status": "loading", "model": MODEL_NAME}
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "diarization": "available" if diarization_pipeline else "unavailable",
    }


def _require_model():
    """确保模型已加载，否则返回 503"""
    if not model_ready.is_set():
        raise HTTPException(503, "模型正在加载中，请稍后重试")
    if model_error or whisper_model is None:
        raise HTTPException(503, f"模型不可用: {model_error}")


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Query("zh", description="语言代码，如 zh, en, ja"),
    diarize: bool = Query(False, description="是否启用说话人分离"),
    _: bool = Depends(verify_token),
):
    """
    转写上传的音频文件

    Headers:
        Authorization: Bearer <your_token>

    Query:
        language: 语言代码 (默认 zh)
        diarize: 是否启用说话人分离 (默认 false)

    Body:
        file: 音频文件 (mp3, wav, m4a, mp4, flac, ogg, webm, wma, aac)

    Response:
        success: bool
        text: 完整转写文本
        segments: [{ start, end, text, speaker? }]
        speakers: 说话人列表 (仅 diarize=true 时)
        language: 检测到的语言
    """
    allowed_ext = {".mp3", ".wav", ".m4a", ".mp4", ".flac", ".ogg", ".webm", ".wma", ".aac"}
    file_ext = Path(file.filename).suffix.lower() if file.filename else ".mp3"

    if file_ext not in allowed_ext:
        raise HTTPException(400, f"不支持的格式: {file_ext}，支持: {', '.join(sorted(allowed_ext))}")

    if diarize and not diarization_pipeline:
        raise HTTPException(400, "说话人分离功能不可用（服务器未配置 HF_TOKEN）")

    tmp_path = None
    try:
        _require_model()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        result = transcribe_audio(tmp_path, language=language, diarize=diarize)

        if not result or not result.get("text"):
            raise HTTPException(500, "转写失败：未识别到语音内容")

        response = {
            "success": True,
            "text": result["text"],
            "segments": result["segments"],
            "language": result["language"],
        }
        if diarize:
            response["speakers"] = result.get("speakers", [])

        return JSONResponse(response)

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
    diarize: bool = Query(False, description="是否启用说话人分离"),
    _: bool = Depends(verify_token),
):
    """
    从 URL 转写音频（支持 Google Drive）

    Headers:
        Authorization: Bearer <your_token>

    Query:
        language: 语言代码 (默认 zh)
        diarize: 是否启用说话人分离 (默认 false)

    Body:
        audio_url: 音频文件 URL
    """
    if diarize and not diarization_pipeline:
        raise HTTPException(400, "说话人分离功能不可用（服务器未配置 HF_TOKEN）")

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

        result = transcribe_audio(tmp_path, language=language, diarize=diarize)

        if not result or not result.get("text"):
            raise HTTPException(500, "转写失败：未识别到语音内容")

        response = {
            "success": True,
            "text": result["text"],
            "segments": result["segments"],
            "language": result["language"],
        }
        if diarize:
            response["speakers"] = result.get("speakers", [])

        return JSONResponse(response)

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
    print(f"   说话人分离: {'启用' if HF_TOKEN else '禁用（需要 HF_TOKEN）'}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
