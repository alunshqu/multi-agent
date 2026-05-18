#!/bin/bash
cd "$(dirname "$0")"

VENV_PYTHON="$(dirname "$0")/.venv/bin/python"
VENV_CHROMA="$(dirname "$0")/.venv/bin/chroma"

# 检查 ChromaDB 服务是否已在运行
if ! curl -s http://localhost:8001/api/v2/heartbeat > /dev/null 2>&1; then
  echo "启动 ChromaDB 服务..."
  nohup "$VENV_CHROMA" run --path ~/.multi_agent/chroma --port 8001 > /tmp/chroma.log 2>&1 &
  # 等待服务就绪（最多 10 秒）
  for i in $(seq 1 10); do
    sleep 1
    if curl -s http://localhost:8001/api/v2/heartbeat > /dev/null 2>&1; then
      echo "ChromaDB 已就绪"
      break
    fi
  done
fi

# 启动主程序
"$VENV_PYTHON" app.py
