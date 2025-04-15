import sys
import os
import ctypes
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal
from mainwindow import MainWindow
from player.snake_player import SnakePlayer
from player.module.screen_capture_thread import ScreenCaptureThread
import time
import cv2


# 创建蛇玩家线程类
class SnakePlayerThread(QThread):
    board_updated = pyqtSignal(
        object, object, float, float, float, float, float, str, object
    )

    def __init__(self, logger=None):
        super().__init__()
        self.snake_player = SnakePlayer(logger)
        self.running = False
        self.hwnd = None
        self.processing = False  # 标记是否正在处理帧

        # 创建截图线程
        self.capture_thread = ScreenCaptureThread(logger)
        self.capture_thread.capture_completed.connect(self.on_capture_completed)

        # 连接信号
        self.snake_player.board_updated.connect(self.board_updated)

    def set_hwnd(self, hwnd):
        """设置游戏窗口句柄"""
        self.hwnd = hwnd
        self.capture_thread.set_hwnd(hwnd)

    def run(self):
        """线程运行函数"""
        self.running = True
        self.capture_thread.start()  # 启动截图线程

        while self.running and not self.isInterruptionRequested():
            try:
                if not self.hwnd:
                    if self.snake_player.logger:
                        self.snake_player.logger.log("等待游戏窗口...")
                    self.msleep(1000)  # 等待窗口时降低检查频率
                    continue

                # 获取最新帧
                frame_data = self.capture_thread.get_latest_frame()
                if frame_data is None:
                    self.msleep(1)  # 等待新帧
                    continue

                hwnd, screen_cv = frame_data
                if hwnd != self.hwnd:  # 窗口已改变，跳过此帧
                    continue

                if self.processing:  # 如果还在处理上一帧，跳过此帧
                    continue

                # 处理帧
                self.processing = True
                process_start = time.time()
                self.snake_player.test_time = time.time()
                self.snake_player.process_frame(screen_cv, self.hwnd)
                process_time = time.time() - process_start
                self.processing = False
                # print(f"处理用时: {process_time*1000:.1f}ms")

            except Exception as e:
                if self.snake_player.logger:
                    self.snake_player.logger.log(f"线程运行错误: {str(e)}")
                self.msleep(1000)  # 发生错误时降低重试频率
                self.processing = False
                continue

            # 控制帧率
            self.msleep(1)  # 约1000fps

    def on_capture_completed(self, hwnd, screen_cv):
        """截图完成回调"""
        pass  # 可以在这里添加额外的处理逻辑

    def stop(self):
        """停止线程"""
        self.running = False
        self.capture_thread.stop()
        self.wait()


def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def run_as_admin():
    script = os.path.abspath(sys.argv[0])
    params = " ".join(sys.argv[1:])
    try:
        if sys.argv[-1] != "asadmin":
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable, f'"{script}" {params} asadmin', None, 1
            )
            sys.exit()
    except:
        pass


if __name__ == "__main__":
    # 检查是否以管理员权限运行
    if not is_admin():
        run_as_admin()
        sys.exit()

    cv2.ocl.setUseOpenCL(True)
    app = QApplication(sys.argv)

    # 创建蛇玩家线程（默认不启动）
    snake_player_thread = SnakePlayerThread()

    # 创建主窗口，传入蛇玩家线程
    window = MainWindow()
    window.snake_player_thread = snake_player_thread

    # 更新游戏卡片中的蛇玩家线程引用
    window.game_widget.snake_player_thread = snake_player_thread

    # 连接信号
    snake_player_thread.board_updated.connect(window.game_widget.on_board_updated)

    # 显示主窗口
    window.show()
    sys.exit(app.exec_())
