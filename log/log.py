from PyQt5.QtWidgets import QTextBrowser
from PyQt5.QtCore import QTimer
from datetime import datetime
from collections import deque


class SnakeLogger:
    def __init__(
        self,
        text_browser: QTextBrowser,
        max_logs=20,
        buffer_size=50,
        update_interval=100,
    ):
        self.text_browser = text_browser
        self.max_logs = max_logs
        # 默认最低等级：DEBUG（全部打印）
        self.log_level = "DEBUG"
        # 定义日志等级优先级
        self.level_priority = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
        # 日志缓冲区
        self.log_buffer = deque(maxlen=buffer_size)
        # 设置定时器进行批量更新
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._flush_buffer)
        self.update_timer.start(update_interval)  # 每100ms更新一次
        self.last_message = None  # 记录上一条消息
        self.repeat_count = 0     # 重复次数计数器

    def _append_text(self, html: str):
        self.log_buffer.append(html)

    def _flush_buffer(self):
        if not self.log_buffer or not self.text_browser:
            return

        # 批量添加日志
        text = "\n".join(self.log_buffer)
        self.text_browser.append(text)
        self.log_buffer.clear()

        # 移动光标到末尾
        cursor = self.text_browser.textCursor()
        cursor.movePosition(cursor.End)
        self.text_browser.setTextCursor(cursor)

        # 控制日志数量
        self._trim_logs()

    def _trim_logs(self):
        # 优化日志清理逻辑
        doc = self.text_browser.document()
        block_count = doc.blockCount()
        if block_count > self.max_logs:
            cursor = self.text_browser.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.movePosition(
                cursor.Down, cursor.KeepAnchor, block_count - self.max_logs
            )
            cursor.removeSelectedText()

    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self._append_text(log_message)
        print(log_message)

    def _log_with_level(self, message: str, level: str, timestamp: str, color: str):
        if message == self.last_message:
            self.repeat_count += 1
            # 更新最后一条日志显示重复次数
            if self.log_buffer and self.text_browser:
                last_msg = self.log_buffer[-1]
                if "(repeated" in last_msg:
                    # 更新已有的重复计数
                    new_msg = last_msg.split("(repeated")[0] + f"(repeated {self.repeat_count})</span><br>"
                    self.log_buffer[-1] = new_msg
                else:
                    # 添加重复计数
                    new_msg = last_msg.replace("</span><br>", f" (repeated {self.repeat_count})</span><br>")
                    self.log_buffer[-1] = new_msg
            return
            
        # 处理新的消息
        self.repeat_count = 0
        self.last_message = message
        log_message = f"[{timestamp}] [{level}] {message}"
        html_msg = f'<span style="color:{color};">{log_message}</span><br>'
        self._append_text(html_msg)

    def debug(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        if self._should_log("DEBUG"):
            self._log_with_level(message, "DEBUG", timestamp, "#555")  # 更深灰

    def info(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        if self._should_log("INFO"):
            self._log_with_level(message, "INFO", timestamp, "#000")  # 黑色

    def warning(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        if self._should_log("WARNING"):
            self._log_with_level(message, "WARNING", timestamp, "#d58512")  # 更深橘黄

    def error(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        if self._should_log("ERROR"):
            self._log_with_level(message, "ERROR", timestamp, "#c9302c")  # 更深红色

    def _should_log(self, level: str) -> bool:
        """判断当前等级是否达到输出标准"""
        return self.level_priority[level] >= self.level_priority.get(self.log_level, 10)
