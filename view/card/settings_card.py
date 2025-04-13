from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt  # 新增的导入
from qfluentwidgets import TitleLabel, ComboBox, Theme
import os
import view.card.theme_manager as theme_manager  # 添加导入


class SettingsCard(QWidget):
    def __init__(self, logger):
        super().__init__()
        self.setObjectName("settingsCard")  # <=== 新增，必须有唯一不为空对象名
        self.logger = logger
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # 日志等级
        self.log_level_label = TitleLabel("日志等级")
        self.log_level_combo = ComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_level_combo.setCurrentText(self.logger.log_level)
        self.log_level_combo.currentTextChanged.connect(self.change_log_level)

        layout.addWidget(self.log_level_label)
        layout.addWidget(self.log_level_combo)

        # 主题切换
        self.theme_label = TitleLabel("主题")
        self.theme_combo = ComboBox()
        self.theme_combo.addItems(["浅色", "深色"])

        if (
            hasattr(theme_manager, "current_theme")
            and theme_manager.current_theme == Theme.DARK
        ):
            self.theme_combo.setCurrentText("深色")
        else:
            self.theme_combo.setCurrentText("浅色")

        self.theme_combo.currentTextChanged.connect(self.change_theme)
        layout.addWidget(self.theme_label)
        layout.addWidget(self.theme_combo)

        # 语言选择
        self.language_label = TitleLabel("语言")
        self.language_combo = ComboBox()
        self.language_combo.addItems(["中文", "English"])
        self.language_combo.setCurrentText("中文")
        self.language_combo.currentTextChanged.connect(self.change_language)

        layout.addWidget(self.language_label)
        layout.addWidget(self.language_combo)

        # 项目目录信息
        repo_path = os.path.abspath(os.path.dirname(__file__))
        self.info_label = QLabel(f"项目路径:\n{repo_path}")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

    def change_log_level(self, level):
        self.logger.log_level = level
        print(f"日志等级切换为：{level}")

    def change_theme(self, theme_text):
        if theme_text == "浅色":
            theme_manager.setTheme(Theme.LIGHT)
        elif theme_text == "深色":
            theme_manager.setTheme(Theme.DARK)

    def change_language(self, lang_text):
        print(f"当前语言: {lang_text}")
