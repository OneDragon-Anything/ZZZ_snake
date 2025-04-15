import threading
from PyQt5.QtCore import QThread, pyqtSignal
from queue import Queue
from player.module.screen_capture import ScreenCapture

class ScreenCaptureThread(QThread):
    # 截图完成信号，发送窗口句柄和图像数据
    capture_completed = pyqtSignal(object, object)
    
    def __init__(self, logger=None):
        super().__init__()
        self.screen_capture = ScreenCapture(logger)
        self.running = False
        self.hwnd = None
        self.capture_queue = Queue(maxsize=1)  # 只保留最新的一帧
        self.lock = threading.Lock()
    
    def set_hwnd(self, hwnd):
        """设置游戏窗口句柄"""
        with self.lock:
            self.hwnd = hwnd
    
    def run(self):
        """线程运行函数"""
        self.running = True
        while self.running and not self.isInterruptionRequested():
            try:
                with self.lock:
                    current_hwnd = self.hwnd
                
                if not current_hwnd:
                    self.msleep(100)  # 等待窗口句柄
                    continue
                
                # 执行截图
                screen_cv = self.screen_capture.capture_screen(current_hwnd)
                
                # 如果截图成功，发送信号
                if screen_cv is not None:
                    # 尝试更新队列，如果队列满则丢弃旧帧
                    try:
                        # 清空队列中的旧帧
                        while not self.capture_queue.empty():
                            self.capture_queue.get_nowait()
                        # 放入新帧
                        self.capture_queue.put_nowait((current_hwnd, screen_cv))
                        # 发送信号
                        self.capture_completed.emit(current_hwnd, screen_cv)
                    except:
                        pass
                
            except Exception as e:
                if hasattr(self.screen_capture, 'log_error'):
                    self.screen_capture.log_error(f"截图线程错误: {str(e)}")
                self.msleep(100)  # 发生错误时降低重试频率
                continue
            
            # 控制帧率
            self.msleep(1)  # 约1000fps
    
    def get_latest_frame(self):
        """获取最新的一帧，如果没有则返回None"""
        try:
            return self.capture_queue.get_nowait()
        except:
            return None
    
    def stop(self):
        """停止线程"""
        self.running = False
        self.wait()