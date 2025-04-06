import logging
from logging.handlers import MemoryHandler
from PyQt5.QtCore import QObject, pyqtSignal, QMetaObject, Qt

class LogSignals(QObject):
    """用于在线程间传递日志信息的信号类"""
    log_signal = pyqtSignal(str)

class SnakeLogger(QObject):
    def __init__(self, text_widget=None):
        """
        初始化日志记录器
        :param text_widget: 可选的文本框部件，用于显示日志
        """
        super().__init__()
        self.logger = logging.getLogger('snake_game')
        self.logger.setLevel(logging.INFO)

        # 创建内存处理器，用于临时存储日志
        self.memory_handler = MemoryHandler(capacity=100, flushLevel=logging.INFO)
        self.logger.addHandler(self.memory_handler)

        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # 设置文本框输出（如果提供了文本框部件）
        self.text_widget = text_widget
        
        # 创建信号对象
        self.signals = LogSignals()
        # 连接信号到更新UI的槽函数
        if self.text_widget:
            self.signals.log_signal.connect(self._update_text_widget, Qt.QueuedConnection)

    def log(self, message, level=logging.INFO):
        """
        记录日志
        :param message: 日志消息
        :param level: 日志级别
        """
        self.logger.log(level, message)

        # 如果设置了文本框，通过信号更新文本框内容
        if self.text_widget:
            # 发送信号到主线程更新UI
            self.signals.log_signal.emit(f"[{logging.getLevelName(level)}] {message}")
    
    def _update_text_widget(self, message):
        """在主线程中更新文本框内容"""
        try:
            current_text = self.text_widget.toPlainText() if hasattr(self.text_widget, 'toPlainText') else self.text_widget.text()
            self.text_widget.setText(f"{current_text}\n{message}")
            # 将光标移动到文档末尾
            cursor = self.text_widget.textCursor()
            cursor.movePosition(cursor.End)
            self.text_widget.setTextCursor(cursor)
            # 确保光标可见
            self.text_widget.ensureCursorVisible()
        except Exception as e:
            print(f"更新文本框时出错: {str(e)}")

    def set_text_widget(self, text_widget):
        """
        设置文本框部件
        :param text_widget: 文本框部件
        """
        self.text_widget = text_widget

    def set_level(self, level):
        """
        设置日志级别
        :param level: 日志级别
        """
        self.logger.setLevel(level)
        self.memory_handler.setLevel(level)

    def clear_logs(self):
        """
        清除日志
        """
        if self.text_widget:
            self.text_widget.setText("")