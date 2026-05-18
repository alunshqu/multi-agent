import asyncio
import threading
import tkinter as tk
from tkinter import scrolledtext
from openai import AsyncOpenAI
from agents.orchestrator import Orchestrator
from core.context import SharedContext
from memory.store import MemoryStore
import config

# 用户反馈关键词
_CORRECTION_WORDS = ("不对", "不是", "错了", "重新", "不对劲", "不行", "不好", "有问题", "不符合")
_SATISFIED_WORDS = ("好的", "完成", "谢谢", "可以", "对的", "没问题", "很好", "棒", "ok", "OK")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI 助手")
        self.geometry("900x680")
        self.configure(bg="#1e1e2e")
        self.resizable(True, True)

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        self._store = MemoryStore()
        if config.OPENAI_API_KEY:
            client = AsyncOpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)
            context = SharedContext()
            self._orchestrator = Orchestrator(client, context, self._store)
        else:
            self._orchestrator = None
        self._last_episode_id: str | None = None
        self._awaiting_feedback = False

        self._build_ui()

    def _build_ui(self):
        # 顶部标题
        header = tk.Frame(self, bg="#181825", pady=10)
        header.pack(fill="x")
        tk.Label(
            header, text="AI 助手", font=("PingFang SC", 18, "bold"),
            bg="#181825", fg="#cdd6f4"
        ).pack()
        tk.Label(
            header, text="告诉我你想做什么，我来帮你完成",
            font=("PingFang SC", 11), bg="#181825", fg="#6c7086"
        ).pack()

        # 对话区域
        chat_frame = tk.Frame(self, bg="#1e1e2e", padx=16, pady=8)
        chat_frame.pack(fill="both", expand=True)

        self._chat = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("PingFang SC", 12),
            bg="#181825", fg="#cdd6f4",
            insertbackground="#cdd6f4",
            relief="flat",
            padx=12, pady=12,
            state="disabled",
            cursor="arrow",
        )
        self._chat.pack(fill="both", expand=True)
        self._chat.tag_config("user_label", foreground="#89b4fa", font=("PingFang SC", 11, "bold"))
        self._chat.tag_config("user_text", foreground="#cdd6f4", font=("PingFang SC", 12))
        self._chat.tag_config("ai_label", foreground="#a6e3a1", font=("PingFang SC", 11, "bold"))
        self._chat.tag_config("ai_text", foreground="#cdd6f4", font=("PingFang SC", 12))
        self._chat.tag_config("error_text", foreground="#f38ba8", font=("PingFang SC", 12))
        self._chat.tag_config("thinking", foreground="#6c7086", font=("PingFang SC", 11, "italic"))

        # 输入区域
        input_frame = tk.Frame(self, bg="#1e1e2e", padx=16, pady=12)
        input_frame.pack(fill="x")

        self._input = tk.Text(
            input_frame,
            height=3,
            font=("PingFang SC", 13),
            bg="#313244", fg="#cdd6f4",
            insertbackground="#cdd6f4",
            relief="flat",
            padx=12, pady=8,
            wrap=tk.WORD,
        )
        self._input.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self._input.bind("<Return>", self._on_enter)
        self._input.bind("<Shift-Return>", lambda e: None)

        self._send_btn = tk.Button(
            input_frame,
            text="发送",
            font=("PingFang SC", 13, "bold"),
            bg="#89b4fa", fg="#1e1e2e",
            activebackground="#74c7ec",
            relief="flat",
            padx=20, pady=8,
            cursor="hand2",
            command=self._send,
        )
        self._send_btn.pack(side="right")

        # 无 API key 提示
        if not self._orchestrator:
            self._append("error_text", "⚠️  未检测到 OPENAI_API_KEY，请设置环境变量后重启。\n\n")

        self._append("thinking", "你好！我可以帮你操作电脑、浏览网页、分析数据、写代码、调用系统……直接告诉我你想做什么。\n\n")

    def _on_enter(self, event):
        if not event.state & 0x1:  # Shift 未按下
            self._send()
            return "break"

    def _send(self):
        text = self._input.get("1.0", tk.END).strip()
        if not text or not self._orchestrator:
            return

        self._input.delete("1.0", tk.END)

        # 检测是否是对上一次任务的反馈
        if self._awaiting_feedback and self._last_episode_id:
            self._detect_and_record_feedback(text)

        self._set_busy(True)
        self._append("user_label", "你\n")
        self._append("user_text", f"{text}\n\n")

        future = asyncio.run_coroutine_threadsafe(
            self._orchestrator.run(text), self._loop
        )
        threading.Thread(target=self._wait_result, args=(future,), daemon=True).start()

    def _detect_and_record_feedback(self, text: str):
        lower = text.lower()
        if any(w in lower for w in _CORRECTION_WORDS):
            feedback = "corrected"
        elif any(w in lower for w in _SATISFIED_WORDS):
            feedback = "satisfied"
        else:
            feedback = "no_response"
        self._store.update_feedback(self._last_episode_id, feedback)
        self._awaiting_feedback = False

    def _wait_result(self, future):
        try:
            result = future.result(timeout=300)
            self.after(0, self._show_result, result, False)
        except Exception as e:
            self.after(0, self._show_result, str(e), True)

    def _show_result(self, text: str, is_error: bool):
        self._set_busy(False)
        tag = "error_text" if is_error else "ai_text"
        self._append("ai_label", "AI 助手\n")
        self._append(tag, f"{text}\n\n")
        # 任务完成后，下一条消息可能是反馈
        if not is_error and self._orchestrator:
            self._last_episode_id = self._orchestrator.last_episode_id
            self._awaiting_feedback = True

    def _set_busy(self, busy: bool):
        if busy:
            self._send_btn.config(state="disabled", text="处理中…")
            self._input.config(state="disabled")
            self._append("thinking", "正在处理，请稍候…\n\n")
        else:
            self._send_btn.config(state="normal", text="发送")
            self._input.config(state="normal")
            self._input.focus()

    def _append(self, tag: str, text: str):
        self._chat.config(state="normal")
        self._chat.insert(tk.END, text, tag)
        self._chat.see(tk.END)
        self._chat.config(state="disabled")


if __name__ == "__main__":
    app = App()
    app.mainloop()
