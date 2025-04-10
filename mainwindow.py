import sys
from PyQt5.QtWidgets import QApplication
from qfluentwidgets import FluentWindow

from view.card.snake_card import SnakeGameCard
from view.card.board_analyzer_test_card import BoardAnalyzerTestCard
from view.card.settings_card import SettingsCard   # 新增导入
from qfluentwidgets import (
    FluentIcon, NavigationItemPosition
)

class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("蛇吃蛇控制器")

        # 成员变量
        self.snake_player_thread = None
        self.game_widget = None
        self.board_analyzer_widget = None
        self.settings_card = None

        self._initInterfaces()
        self._initNavigation()

    def _initInterfaces(self):
        """初始化界面"""
        self.game_widget = SnakeGameCard(snake_player_thread=self.snake_player_thread)

        self.board_analyzer_widget = BoardAnalyzerTestCard()

        # 传递日志logger给设置界面
        self.settings_card = SettingsCard(self.game_widget.logger)

        self.addSubInterface(
            self.game_widget,
            FluentIcon.LIBRARY,
            'game',
            position=NavigationItemPosition.TOP
        )
        self.addSubInterface(
            self.board_analyzer_widget,
            FluentIcon.ALBUM,
            'board_analyzer',
            position=NavigationItemPosition.TOP
        )
        self.addSubInterface(
            self.settings_card,
            FluentIcon.SETTING,
            'settings',
            position=NavigationItemPosition.BOTTOM
        )

    def _initNavigation(self):
        """初始化导航栏及其行为"""
        self.navigationInterface.setExpandWidth(200)
        self.resize(500, 500)
        self.titleBar.raise_()
        # 绑定导航切换信号
        self.stackedWidget.currentChanged.connect(self.onNavigationChanged)

    def onNavigationChanged(self, index):
        """处理导航栏切换事件"""
        current_widget = self.stackedWidget.widget(index)

        # 使用switchTo方法切换界面
        self.switchTo(current_widget)

        # 切换非游戏界面时，停止蛇线程
        if current_widget != self.game_widget and self.snake_player_thread:
            if self.game_widget.btn_snake_player.isChecked():
                self.game_widget.btn_snake_player.setChecked(False)
                # onSnakePlayerToggled会被自动调用，不需要手动停止线程

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())