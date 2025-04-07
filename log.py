from PyQt5.QtWidgets import QTextBrowser
from datetime import datetime

class SnakeLogger:
    def __init__(self, text_browser: QTextBrowser):
        self.text_browser = text_browser

    def log(self, message: str):
        """记录日志信息"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_message = f'[{timestamp}] {message}'
        self.text_browser.append(log_message)