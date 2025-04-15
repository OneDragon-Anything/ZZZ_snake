import logging
from datetime import datetime

class LoggingMixin:
    """
    提供统一的日志记录方法的混入类
    所有重构后的类都应该继承这个类来获得统一的日志记录功能
    """
    
    def __init__(self, logger=None):
        self.logger = logger
    
    def _format_message(self, message, level="INFO"):
        """格式化日志消息"""
        return f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] [{level}] [{self.__class__.__name__}] {message}"
        
    def log_debug(self, message):
        """记录调试信息"""
        if self.logger:
            self.logger.debug(message)
        else:
            print(self._format_message(message, "DEBUG"))
            
    def log_info(self, message):
        """记录一般信息"""
        if self.logger:
            self.logger.info(message)
        else:
            print(self._format_message(message, "INFO"))
            
    def log_warning(self, message):
        """记录警告信息"""
        if self.logger:
            self.logger.warning(message)
        else:
            print(self._format_message(message, "WARNING"))
            
    def log_error(self, message, exception=None):
        """记录错误信息"""
        if exception:
            message = f"{message}: {exception}"
            
        if self.logger:
            self.logger.error(message)
        else:
            print(self._format_message(message, "ERROR"))