import sys
import ctypes

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from qfluentwidgets import FluentWindow

from snake_card import SnakeGameCard


class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("蛇吃蛇控制器")

        # 创建主部件
        self.game_widget = SnakeGameCard()
        self.addSubInterface(self.game_widget, 'game', '游戏控制')

        # 初始化导航栏
        self.navigationInterface.setExpandWidth(200)
        self.setMinimumSize(600, 600)


def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


if __name__ == "__main__":
    if not is_admin():
        # 确保完全无控制台
        ctypes.windll.kernel32.FreeConsole()
        # 使用pythonw.exe路径重新运行
        pythonw_path = sys.executable.replace('python.exe', 'pythonw.exe')
        ctypes.windll.shell32.ShellExecuteW(None, "runas", pythonw_path, " ".join(sys.argv), None, 1)
        sys.exit()
    
    # 正常启动程序
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())