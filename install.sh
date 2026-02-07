#!/bin/bash
# Paraformer API 一键部署脚本

set -e

PORT=8000

echo "=== Paraformer API 一键部署 ==="

# 输入 Token
read -p "请输入 API Token: " API_TOKEN
if [ -z "$API_TOKEN" ]; then
    echo "错误: Token 不能为空"
    exit 1
fi

# 1. 安装依赖
echo ">>> 安装 Python 依赖..."
pip install fastapi uvicorn funasr modelscope torch torchaudio requests -q

# 2. 下载 API 文件
echo ">>> 下载 API 服务..."
curl -sSL https://raw.githubusercontent.com/toller892/paraformer-api-server/main/paraformer_api.py -o paraformer_api.py

# 3. 启动服务
echo ">>> 启动服务（首次会下载模型约 1GB）..."
API_TOKEN=$API_TOKEN nohup python paraformer_api.py > paraformer.log 2>&1 &

echo ""
echo "=== 部署完成 ==="
echo "端口: $PORT"
echo "日志: tail -f paraformer.log"
echo ""
echo "验证: curl http://localhost:$PORT/health"
