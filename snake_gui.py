import sys
import ctypes

import cv2
import numpy as np
import pyautogui
import win32con
import win32gui
from PIL import Image, ImageGrab
from PyQt5.QtCore import Qt, QEasingCurve, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFrame, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget
)
from qfluentwidgets import (
    BodyLabel, CaptionLabel, FluentWindow, ImageLabel,
    PrimaryPushButton, PushButton, ScrollArea,
    SingleDirectionScrollArea, SmoothScrollArea,
    TextBrowser, ToolTipFilter, SwitchButton
)
from qfluentwidgets import FlowLayout, HorizontalPipsPager, PixmapLabel
from qfluentwidgets import PipsScrollButtonDisplayMode, VerticalPipsPager

from log import SnakeLogger
from snake_analyzer import SnakeAnalyzer


class GameWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('GameWidget')

        layout = FlowLayout(self, needAni=True)

        # 自定义动画效果
        # 设置动画持续时间为 250 毫秒，缓动曲线为 QEasingCurve.OutQuad
        layout.setAnimation(250, QEasingCurve.OutQuad)

        # 设置布局的边距 (上, 左, 下, 右)
        layout.setContentsMargins(30, 30, 30, 30)
        # 设置子控件之间的垂直间距
        layout.setVerticalSpacing(20)
        # 设置子控件之间的水平间距
        layout.setHorizontalSpacing(50)

        # 游戏画面预览
        self.screen_label = ImageLabel()
        self.screen_label.scaledToWidth(775)
        self.screen_label.setBorderRadius(8, 8, 8, 8)

        self.w = ScrollArea()
        self.w.horizontalScrollBar().setValue(0)
        self.w.setWidget(self.screen_label)
        self.w.setFixedSize(775, 430)

        # 开始运行按钮
        self.btn_start = PrimaryPushButton("开始运行")
        self.btn_start.clicked.connect(self.start_running)

        # 停止运行按钮
        self.btn_stop = PrimaryPushButton("停止运行")
        self.btn_stop.clicked.connect(self.stop_running)
        self.btn_stop.setEnabled(False)

        # 分析结果显示开关
        self.btn_show_analysis = SwitchButton(parent=self)
        self.btn_show_analysis.setChecked(True)
        self.btn_show_analysis.checkedChanged.connect(self.onAnalysisToggled)

        # 状态显示区域
        self.status_label = CaptionLabel("状态: 等待开始运行")
        self.status_label.setAlignment(Qt.AlignCenter)

        # 日志文本框
        self.log_text = TextBrowser()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        self.log_text.setMinimumWidth(500)

        # 添加用于显示游戏状态的标签
        self.game_state_label = BodyLabel("当前状态: 初始化")  # 初始状态
        self.game_state_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.screen_label)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.btn_show_analysis)
        layout.addWidget(self.log_text)
        layout.addWidget(self.status_label)
        layout.addWidget(self.game_state_label)

        # 定时器更新画面
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_screen)

        # 当前选择的窗口句柄
        self.selected_hwnd = None

        # 初始化 logger
        self.logger = SnakeLogger(self.log_text)
        # 创建 SnakeAnalyzer 的实例，并将 logger 传递给它
        self.snake_analyzer = SnakeAnalyzer(self.logger)

    def update_screen(self):
        """更新游戏画面预览并使用 OpenCV 进行处理"""
        # 初始化默认屏幕变量
        screen = None

        # 使用用户选择的窗口或默认查找'绝区零'窗口
        hwnd = self.selected_hwnd if self.selected_hwnd else win32gui.FindWindow(None, '绝区零')
        if hwnd:
            try:
                # 获取窗口客户区位置和大小
                client_left, client_top, client_right, client_bottom = win32gui.GetClientRect(hwnd)
                # 转换为屏幕坐标
                left, top = win32gui.ClientToScreen(hwnd, (client_left, client_top))
                right, bottom = win32gui.ClientToScreen(hwnd, (client_right, client_bottom))
                # 使用ImageGrab捕获窗口内容
                screen = ImageGrab.grab(bbox=(left, top, right, bottom))
                # 缩放为1920x1080分辨率
                screen = screen.resize((1920, 1080))
                # 裁剪指定区域
                screen = screen.crop((485, 202, 1434, 1029))

                # 检查窗口是否在最前
                if win32gui.GetForegroundWindow() == hwnd:
                    try:
                        # 将 PIL Image 转换为 OpenCV 格式 (NumPy 数组)
                        screen_cv = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)

                        # 调用 SnakeAnalyzer 的 analyze_frame 方法
                        game_state, path, special_cells = self.snake_analyzer.analyze_frame(screen_cv)

                        # 更新 UI 上的状态标签
                        self.game_state_label.setText(f"当前状态: {game_state}")

                        # 如果开启了分析结果显示
                        if self.btn_show_analysis.isChecked():
                            # 计算格子大小
                            h, w = screen_cv.shape[:2]
                            grid_h, grid_w = 25, 29
                            cell_h = h / grid_h
                            cell_w = w / grid_w

                            # 绘制网格线
                            for i in range(grid_h + 1):
                                y = int(i * cell_h)
                                cv2.line(screen_cv, (0, y), (w, y), (255, 255, 255), 1)  # 白色横线
                            for j in range(grid_w + 1):
                                x = int(j * cell_w)
                                cv2.line(screen_cv, (x, 0), (x, h), (255, 255, 255), 1)  # 白色竖线

                            # 绘制路径
                            if path and len(path) > 1:
                                for i in range(len(path)-1):
                                    start_x = int(path[i][0] * cell_w + cell_w / 2)
                                    start_y = int(path[i][1] * cell_h + cell_h / 2)
                                    end_x = int(path[i+1][0] * cell_w + cell_w / 2)
                                    end_y = int(path[i+1][1] * cell_h + cell_h / 2)
                                    cv2.line(screen_cv, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)  # 蓝色路径

                            # 绘制特殊格子和中心点
                            for (x, y), cell_type in special_cells.items():
                                left = int(x * cell_w)
                                top = int(y * cell_h)
                                right = int(left + cell_w)
                                bottom = int(top + cell_h)
                                center_x = int(left + cell_w / 2)
                                center_y = int(top + cell_h / 2)
                                
                                if cell_type == 'speed_boost':
                                    color = (0, 255, 0)  # 绿色
                                elif cell_type == 'score_boost':
                                    color = (255, 165, 0)  # 橙色
                                elif cell_type == 'own_head':
                                    color = (0, 255, 255)  # 黄色
                                elif cell_type == 'own_body':
                                    color = (255, 255, 255)  # 白色
                                elif cell_type == 'enemy_head':
                                    color = (255, 0, 0)  # 红色
                                elif cell_type == 'enemy_body':
                                    color = (128, 0, 0)  # 深红色
                                else:
                                    continue
                                
                                cv2.rectangle(screen_cv, (left, top), (right, bottom), color, 2)
                                cv2.circle(screen_cv, (center_x, center_y), 4, color, -1)  # 在格子中心绘制实心圆点

                        # 将处理后的图像转换回RGB格式
                        screen_cv = cv2.cvtColor(screen_cv, cv2.COLOR_BGR2RGB)
                        # 将OpenCV图像转换回PIL格式
                        screen = Image.fromarray(screen_cv)

                    except Exception as e:
                        self.status_label.setText(f"分析图像错误: {str(e)}")
                        self.screen_label.clear()
                        return
            except Exception as e:
                self.status_label.setText(f"捕获窗口错误: {str(e)}")
                self.screen_label.clear()
                return

        if screen is None:
            return

        try:
            # 总是显示画面
            screen_np = np.array(screen)
            height, width, channel = screen_np.shape
            bytes_per_line = 3 * width
            q_img = QImage(screen_np.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.screen_label.setPixmap(QPixmap.fromImage(q_img))

        except Exception as e:
            self.status_label.setText(f"处理图像错误: {str(e)}")
            self.screen_label.clear()

    def start_running(self):
        """开始运行"""
        # 自动查找标题为'绝区零'的窗口
        hwnd = win32gui.FindWindow(None, '绝区零')
        if hwnd:
            self.selected_hwnd = hwnd
            self.status_label.setText("状态: 已找到'绝区零'窗口")
            # 启动定时器
            self.timer.start(33)
            # 更新按钮状态
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
        else:
            self.status_label.setText("状态: 未找到'绝区零'窗口")

    def stop_running(self):
        """停止运行"""
        self.timer.stop()
        self.selected_hwnd = None
        self.status_label.setText("状态: 已停止运行")
        # 更新按钮状态
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def onAnalysisToggled(self, isChecked: bool):
        """处理分析结果显示开关的状态变化"""
        text = '开启分析' if isChecked else '关闭分析'
        self.btn_show_analysis.setText(text)

    def select_window(self):
        """选择要捕获的窗口"""
        # 自动查找标题为'绝区零'的窗口
        hwnd = win32gui.FindWindow(None, '绝区零')
        if hwnd:
            self.selected_hwnd = hwnd
            self.status_label.setText("状态: 已找到'绝区零'窗口")
        else:
            self.status_label.setText("状态: 未找到'绝区零'窗口")

        QTimer.singleShot(150, lambda: self.window().adjustSize())


class SnakeGUI(FluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("蛇吃蛇控制器")

        # 创建主部件
        self.game_widget = GameWidget()
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
    window = SnakeGUI()
    window.show()
    sys.exit(app.exec_())