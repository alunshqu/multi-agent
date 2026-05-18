#!/bin/bash
cd "$(dirname "$0")"
if [ ! -f .env ]; then
  cp .env.example .env
  open -e .env
  osascript -e 'display dialog "请在打开的 .env 文件中填入你的 ANTHROPIC_API_KEY，保存后重新双击启动。" buttons {"好的"} default button 1'
  exit 0
fi
python3 app.py
