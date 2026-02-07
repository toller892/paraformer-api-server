#!/usr/bin/env python3
"""
Paraformer API 服务
支持语音转写 + 说话人分离，Token 鉴权
"""
import os
import re
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
PORT = int(os.getenv("PORT", 8000))

# ============ 初始化模型 ============
print("正在加载 Paraformer 语音识别模型...")
asr_model = AutoModel(model="paraformer-zh")
print("Paraformer 模型加载完成！")

print("正在加载 CAM++ 说话人分离模型...")
spk_model = AutoModel(model="cam++", model_revision="master")
print("CAM++ 模型加载完成！")

# ============ FastAPI 应用 ============
app = FastAPI(
    title="Paraformer ASR API",
    description="语音转写 API 服务（支持说话人分离）",
    version="2.0.0"
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


def transcribe_audio(file_path: str, diarize: bool = False) -> dict:
    """转写音频，可选说话人分离"""
    
    if not diarize:
        # 纯转写模式
        result = asr_model.generate(input=file_path)
        if not result:
            return None
        text = ""
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                text = result[0].get("text", "")
            else:
                text = str(result[0])
        return {"text": text}
    
    else:
        # 说话人分离模式
        # 1. 先做语音识别（带时间戳）
        asr_result = asr_model.generate(
            input=file_path,
            batch_size_s=300,
            sentence_timestamp=True
        )
        
        if not asr_result:
            return None
        
        # 2. 做说话人分离
        spk_result = spk_model.generate(input=file_path)
        
        # 3. 合并结果
        utterances = []
        
        if isinstance(asr_result, list) and len(asr_result) > 0:
            asr_data = asr_result[0]
            
            # 获取句子级别结果
            sentences = asr_data.get("sentence_info", [])
            if not sentences and "text" in asr_data:
                # 没有句子分割，返回整体
                utterances.append({
                    "speaker": "speaker_0",
                    "start": 0,
                    "end": 0,
                    "text": asr_data["text"]
                })
            else:
                # 有句子分割
                for i, sent in enumerate(sentences):
                    spk_id = f"speaker_{sent.get('spk', 0)}" if 'spk' in sent else f"speaker_{i % 2}"
                    utterances.append({
                        "speaker": spk_id,
                        "start": sent.get("start", 0) / 1000,  # 转换为秒
                        "end": sent.get("end", 0) / 1000,
                        "text": sent.get("text", "")
                    })
        
        full_text = " ".join([u["text"] for u in utterances])
        return {"text": full_text, "utterances": utterances}


# ============ API 端点 ============
@app.get("/")
async def root():
    return {"status": "ok", "service": "Paraformer ASR API", "version": "2.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    diarize: bool = Query(False, description="是否启用说话人分离"),
    _: bool = Depends(verify_token)
):
    """
    转写上传的音频文件
    
    Headers:
        Authorization: Bearer <your_token>
    
    Query:
        diarize: 是否启用说话人分离 (默认 false)
    
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
        
        result = transcribe_audio(tmp_path, diarize=diarize)
        os.unlink(tmp_path)
        
        if not result:
            raise HTTPException(500, "转写失败")
        
        response = {"success": True, "text": result["text"]}
        if "utterances" in result:
            response["utterances"] = result["utterances"]
        
        return JSONResponse(response)
    
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(500, f"处理错误: {str(e)}")


@app.post("/transcribe/url")
async def transcribe_url(
    audio_url: str,
    diarize: bool = Query(False, description="是否启用说话人分离"),
    _: bool = Depends(verify_token)
):
    """
    从 URL 转写音频（支持 Google Drive）
    
    Headers:
        Authorization: Bearer <your_token>
    
    Query:
        diarize: 是否启用说话人分离 (默认 false)
    
    Body:
        audio_url: 音频文件 URL
    """
    try:
        download_url = convert_gdrive_url(audio_url)
        
        response = requests.get(download_url, timeout=300)
        response.raise_for_status()
        
        file_ext = Path(audio_url).suffix.lower()
        if not file_ext or file_ext not in {'.mp3', '.wav', '.m4a', '.mp4', '.flac', '.ogg'}:
            file_ext = '.mp3'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        
        result = transcribe_audio(tmp_path, diarize=diarize)
        os.unlink(tmp_path)
        
        if not result:
            raise HTTPException(500, "转写失败")
        
        resp = {"success": True, "text": result["text"]}
        if "utterances" in result:
            resp["utterances"] = result["utterances"]
        
        return JSONResponse(resp)
    
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
