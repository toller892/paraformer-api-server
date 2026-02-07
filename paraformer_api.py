#!/usr/bin/env python3
"""
Paraformer API 服务
纯净版 - 仅提供语音转写功能，支持 Token 鉴权
"""
import os
import sys
import re
import tempfile
from pathlib import Path
from typing import Optional

# 抑制模型下载的频繁日志
os.environ["MODELSCOPE_DOWNLOAD_PROGRESS"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import requests
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# 自定义进度回调
class DownloadProgress:
    def __init__(self):
        self.last_percent = -1
    
    def __call__(self, current, total):
        if total > 0:
            percent = int(current * 100 / total)
            if percent != self.last_percent and percent % 10 == 0:
                print(f"模型下载进度: {percent}%", flush=True)
                self.last_percent = percent

print("正在加载 Paraformer 模型（首次需下载约 1GB）...", flush=True)

from funasr import AutoModel
model = AutoModel(model="paraformer-zh")
print("模型加载完成！", flush=True)

# ============ 配置 ============
API_TOKEN = os.getenv("API_TOKEN", "")  # 必须设置环境变量
PORT = int(os.getenv("PORT", 8000))

# ============ FastAPI 应用 ============
app = FastAPI(
    title="Paraformer ASR API",
    description="语音转写 API 服务",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Token 验证 ============
def verify_token(authorization: Optional[str] = Header(None)):
    """验证 Bearer Token"""
    if not API_TOKEN:
        raise HTTPException(500, "服务器未配置 API_TOKEN")
    
    if not authorization:
        raise HTTPException(401, "缺少 Authorization 头")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Authorization 格式错误，需要 Bearer Token")
    
    token = authorization[7:]
    if token != API_TOKEN:
        raise HTTPException(403, "Token 无效")
    
    return True


# ============ 工具函数 ============
def convert_gdrive_url(url: str) -> str:
    """将 Google Drive 分享链接转换为直接下载链接"""
    if 'export=download' in url:
        return url
    
    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'[?&]id=([a-zA-Z0-9_-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            file_id = match.group(1)
            return f"https://drive.google.com/uc?export=download&id={file_id}"
    
    return url


def transcribe_audio(file_path: str) -> dict:
    """调用 Paraformer 转写音频"""
    result = model.generate(input=file_path)
    
    if not result:
        return None
    
    # 提取文本
    text = ""
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], dict):
            text = result[0].get("text", "")
        else:
            text = str(result[0])
    
    return {"text": text}


# ============ API 端点 ============
@app.get("/")
async def root():
    """健康检查"""
    return {"status": "ok", "service": "Paraformer ASR API"}


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    _: bool = Depends(verify_token)
):
    """
    转写上传的音频文件
    
    Headers:
        Authorization: Bearer <your_token>
    
    Body:
        file: 音频文件 (mp3, wav, m4a, mp4, flac, ogg)
    """
    allowed_ext = {'.mp3', '.wav', '.m4a', '.mp4', '.flac', '.ogg'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_ext:
        raise HTTPException(400, f"不支持的格式: {file_ext}")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        result = transcribe_audio(tmp_path)
        os.unlink(tmp_path)
        
        if not result:
            raise HTTPException(500, "转写失败")
        
        return JSONResponse({"success": True, "text": result["text"]})
    
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(500, f"处理错误: {str(e)}")


@app.post("/transcribe/url")
async def transcribe_url(
    audio_url: str,
    _: bool = Depends(verify_token)
):
    """
    从 URL 转写音频（支持 Google Drive）
    
    Headers:
        Authorization: Bearer <your_token>
    
    Body:
        audio_url: 音频文件 URL
    """
    try:
        download_url = convert_gdrive_url(audio_url)
        
        response = requests.get(download_url, timeout=300)
        response.raise_for_status()
        
        # 推断扩展名
        file_ext = Path(audio_url).suffix.lower()
        if not file_ext or file_ext not in {'.mp3', '.wav', '.m4a', '.mp4', '.flac', '.ogg'}:
            file_ext = '.mp3'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        
        result = transcribe_audio(tmp_path)
        os.unlink(tmp_path)
        
        if not result:
            raise HTTPException(500, "转写失败")
        
        return JSONResponse({"success": True, "text": result["text"]})
    
    except requests.RequestException as e:
        raise HTTPException(400, f"下载失败: {str(e)}")
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(500, f"处理错误: {str(e)}")


if __name__ == "__main__":
    if not API_TOKEN:
        print("⚠️  警告: 未设置 API_TOKEN 环境变量！")
        print("   请使用: API_TOKEN=your_secret_token python paraformer_api.py")
        exit(1)
    
    print(f"服务启动在 http://0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
