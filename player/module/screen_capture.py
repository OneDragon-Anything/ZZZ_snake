import cv2
import numpy as np
import win32gui
import mss
import threading
from player.module.logging_mixin import LoggingMixin

class ScreenCapture(LoggingMixin):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.sct = None
        self.init_thread_id = None
        self.last_window_hwnd = None
        self.last_window_rect = None
        self.lock = threading.Lock()

    def __del__(self):
        with self.lock:
            if self.sct:
                self.sct.close()

    def _init_screen_capture(self) -> bool:
        current_thread_id = threading.get_ident()
        with self.lock:
            if self.sct is None or self.init_thread_id != current_thread_id:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        # 确保旧的实例被正确清理
                        if self.sct:
                            try:
                                self.sct.close()
                            except:
                                pass
                            self.sct = None
                        
                        # 强制等待一小段时间确保资源释放
                        if attempt > 0:
                            import time
                            time.sleep(0.1)
                        
                        # 创建新实例
                        self.sct = mss.mss()
                        self.init_thread_id = current_thread_id
                        return True
                        
                    except Exception as e:
                        self.log_error(f"初始化屏幕捕获失败(尝试 {attempt + 1}/{max_retries}): {str(e)}")
                        if attempt == max_retries - 1:
                            return False
            return True

    def capture_screen(self, hwnd) -> "np.ndarray|None":
        if not self._validate_window(hwnd):
            return None

        try:
            if hwnd != self.last_window_hwnd and not self._update_window_rect(hwnd):
                return None

            if not self._init_screen_capture():
                return None

            with self.lock:
                sct_img = self.sct.grab(self.last_window_rect)

            if not sct_img or not hasattr(sct_img, 'size'):
                return None

            img = np.asarray(sct_img)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        except Exception as e:
            self.log_error(f"画面捕获错误: {str(e)}")
            return None

    def _validate_window(self, hwnd) -> bool:
        if not hwnd or not win32gui.IsWindow(hwnd):
            return False

        if not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
            return False

        return True

    def _update_window_rect(self, hwnd) -> bool:
        try:
            cl, ct, cr, cb = win32gui.GetClientRect(hwnd)
            if cr <= 0 or cb <= 0:
                return False

            left, top = win32gui.ClientToScreen(hwnd, (cl, ct))
            right, bottom = win32gui.ClientToScreen(hwnd, (cr, cb))

            width = right - left
            height = bottom - top

            if width <= 0 or height <= 0 or abs(width / height - 16 / 9) > 0.01:
                return False

            x1 = int(left + width * 485 / 1920)
            y1 = int(top + height * 203 / 1080)
            x2 = int(left + width * 1434 / 1920)
            y2 = int(top + height * 1028 / 1080)

            self.last_window_rect = {
                "left": x1,
                "top": y1,
                "width": x2 - x1,
                "height": y2 - y1,
            }
            self.last_window_hwnd = hwnd
            return True

        except Exception:
            return False