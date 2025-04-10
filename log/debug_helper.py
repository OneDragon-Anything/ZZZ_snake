import os
import threading
import time
import cv2

class DebugHelper:
    _log_file_path = os.path.join(os.path.dirname(__file__), '..', 'debug', 'log.txt')
    _image_dir = os.path.join(os.path.dirname(__file__), '..', 'debug', 'images')
    _lock = threading.Lock()

    @staticmethod
    def log_message(message, file_name=None):
        """将调试日志写入指定文件，默认为 debug/log.txt"""
        with DebugHelper._lock:
            try:
                if file_name:
                    log_path = os.path.join(os.path.dirname(DebugHelper._log_file_path), file_name)
                else:
                    log_path = DebugHelper._log_file_path
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f"[{timestamp}] {message}\n")
            except Exception as e:
                print(f"写入调试日志失败: {e}")

    @staticmethod
    def save_image(image, prefix='debug', file_name=None):
        """保存OpenCV格式的图像为png，支持自定义文件名"""
        with DebugHelper._lock:
            try:
                os.makedirs(DebugHelper._image_dir, exist_ok=True)
                if file_name:
                    file_path = os.path.join(DebugHelper._image_dir, file_name)
                    if not file_path.lower().endswith('.png'):
                        file_path += '.png'
                else:
                    timestamp = int(time.time() * 1000)
                    filename = f"{prefix}_{timestamp}.png"
                    file_path = os.path.join(DebugHelper._image_dir, filename)
                cv2.imwrite(file_path, image)
                return file_path
            except Exception as e:
                print(f"保存调试图片失败: {e}")
                return None