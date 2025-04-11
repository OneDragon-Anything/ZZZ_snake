import sys
import os
import ctypes
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal
from mainwindow import MainWindow
from player.snake_player import SnakePlayer
import time
# 创建蛇玩家线程类
class SnakePlayerThread(QThread):
    board_updated = pyqtSignal(object, object, float, float, float, float, float, str, object)
    
    def __init__(self, logger=None):
        super().__init__()
        self.snake_player = SnakePlayer(logger)
        self.running = False
        self.hwnd = None
        
        # 连接信号
        self.snake_player.board_updated.connect(self.board_updated)
    
    def set_hwnd(self, hwnd):
        """设置游戏窗口句柄"""
        self.hwnd = hwnd
    
    def run(self):
        """线程运行函数"""
        self.running = True
        error_count = 0  # 添加错误计数器
        while self.running and not self.isInterruptionRequested():
            
            try:
                if not self.hwnd:
                    if self.snake_player.logger:
                        self.snake_player.logger.log("等待游戏窗口...")
                    self.msleep(1000)  # 等待窗口时降低检查频率
                    continue
                
                # 捕获屏幕
                self.snake_player.test_time=time.time()
                screen_cv = self.snake_player.capture_screen(self.hwnd)
                if screen_cv is None:
                    error_count += 1
                    if error_count >= 3 and self.snake_player.logger:  # 连续3次失败才记录
                        self.snake_player.logger.log("画面捕获失败，请检查游戏窗口是否正常")
                        error_count = 0  # 重置计数器
                    self.msleep(500)  # 捕获失败时降低重试频率
                    continue
                
                # 处理帧
                error_count = 0  # 成功后重置错误计数
                
                self.snake_player.process_frame(screen_cv, self.hwnd)
            except Exception as e:
                if self.snake_player.logger:
                    self.snake_player.logger.log(f"线程运行错误: {str(e)}")
                self.msleep(1000)  # 发生错误时降低重试频率
                continue
                
            # 控制帧率
            self.msleep(2)  # 约20帧每秒
    
    def stop(self):
        """停止线程"""
        self.running = False
        self.wait()


def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def run_as_admin():
    script = os.path.abspath(sys.argv[0])
    params = ' '.join(sys.argv[1:])
    try:
        if sys.argv[-1] != 'asadmin':
            ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{script}" {params} asadmin', None, 1)
            sys.exit()
    except:
        pass

if __name__ == "__main__":
    # 检查是否以管理员权限运行
    if not is_admin():
        run_as_admin()
        sys.exit()
        
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