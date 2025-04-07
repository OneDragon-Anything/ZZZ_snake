import cv2
import numpy as np
import win32gui
from PIL import Image, ImageGrab
from PyQt5.QtCore import Qt, QEasingCurve, QTimer
from PyQt5.QtGui import QImage, QPixmap
import time
from PyQt5.QtWidgets import QFrame, QHBoxLayout
from qfluentwidgets import (
    BodyLabel, CaptionLabel, ImageLabel,
    PrimaryPushButton, ScrollArea,
    TextBrowser, SwitchButton, ComboBox,
    InfoBadge
)
from qfluentwidgets import FlowLayout

from log import SnakeLogger
from image_analyzer import ImageAnalyzer
from path_calculator import PathCalculator
from snake_controller import SnakeController
from map_drawer import MapDrawer
from qfluentwidgets import TransparentDropDownPushButton, RoundMenu, Action, FluentIcon, setFont


class SnakeGameCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('SnakeGameCard')

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

        # 画面刷新开关
        self.btn_show_screen = SwitchButton(parent=self)
        self.btn_show_screen.setChecked(True)
        self.btn_show_screen.setText('开启画面刷新')
        self.btn_show_screen.checkedChanged.connect(self.onScreenToggled)
        
        # 路径计算频率控制
        self.path_calc_label = CaptionLabel("路径计算频率:")
        self.path_calc_combo = ComboBox()
        self.path_calc_combo.addItems(["高 (每帧)", "中 (100ms)", "低 (200ms)"])
        self.path_calc_combo.setCurrentIndex(1)  # 默认选择中等频率
        self.path_calc_combo.currentIndexChanged.connect(self.onPathCalcFrequencyChanged)



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

        # 添加性能计时显示栏
        self.performance_layout = QHBoxLayout()
        self.convert_time_badge = InfoBadge.custom("图像转换: 0.000s", "#0066cc", "#ffffff")
        self.analyze_time_badge = InfoBadge.custom("棋盘分析: 0.000s", "#00cc66", "#ffffff")
        self.path_calc_time_badge = InfoBadge.custom("路径计算: 0.000s", "#ffcc00", "#000000")
        self.draw_time_badge = InfoBadge.custom("绘制: 0.000s", "#cc66ff", "#ffffff")
        self.snake_direction_badge = InfoBadge.custom("方向: 无", "#ff6600", "#ffffff")
        
        self.performance_layout.addWidget(self.convert_time_badge)
        self.performance_layout.addWidget(self.analyze_time_badge)
        self.performance_layout.addWidget(self.path_calc_time_badge)
        self.performance_layout.addWidget(self.draw_time_badge)
        self.performance_layout.addWidget(self.snake_direction_badge)
        self.performance_layout.setSpacing(10)
        self.performance_layout.setAlignment(Qt.AlignCenter)

        # 创建一个容器来包含性能计时显示栏
        self.performance_container = QFrame()
        self.performance_container.setLayout(self.performance_layout)

        layout.addWidget(self.screen_label)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.btn_show_analysis)
        layout.addWidget(self.btn_show_screen)
        layout.addWidget(self.path_calc_label)
        layout.addWidget(self.path_calc_combo)

        layout.addWidget(self.performance_container)
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
        # 创建各个模块的实例，并将 logger 传递给它们
        self.image_analyzer = ImageAnalyzer(self.logger)
        self.path_calculator = PathCalculator(self.logger)
        self.snake_controller = SnakeController(self.logger)
        self.map_drawer = MapDrawer()

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
                        # 性能计时 - 开始
                        start_time = time.time()
                        
                        # 将 PIL Image 转换为 OpenCV 格式 (NumPy 数组)
                        screen_cv = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
                        convert_time = time.time() - start_time

                        # 1. 调用图像分析模块分析画面
                        analyze_start = time.time()
                        try:
                            game_state, board_state, special_cells = self.image_analyzer.analyze_frame(screen_cv)
                        except Exception as e:
                            self.logger.log(f"图像分析错误: {str(e)}")
                            game_state, board_state, special_cells = None, None, {}
                            analyze_time = 0
                        analyze_time = time.time() - analyze_start
                        
                        # 2. 如果游戏正在运行，计算路径
                        path = []
                        path_calc_start = time.time()
                        if game_state == "running" and board_state:
                            try:
                                # 获取蛇头位置
                                head_pos = None
                                for r in range(len(board_state)):
                                    for c in range(len(board_state[r])):
                                        if board_state[r][c] == 'own_head':
                                            head_pos = (c, r)
                                            break
                                    if head_pos:
                                        break
                                
                                # 计算路径
                                if head_pos:
                                    path = self.path_calculator.calculate_path(board_state, head_pos)
                            except Exception as e:
                                self.logger.log(f"路径计算错误: {str(e)}")
                                path = []
                                path_calc_time = 0
                        path_calc_time = time.time() - path_calc_start

                        # 3. 如果游戏正在运行，控制蛇移动
                        try:
                            if game_state == "running" and board_state:
                                # 设置游戏窗口
                                self.snake_controller.set_game_window(self.selected_hwnd)
                                
                                # 根据路径确定下一步移动并控制蛇
                                next_move = self.snake_controller.determine_next_move(board_state, path)
                                if next_move:
                                    self.snake_controller.control_snake(next_move)
                            # 如果游戏已结束，尝试点击窗口中心
                            elif game_state in ["game_over"]:
                                self.snake_controller.set_game_window(self.selected_hwnd)
                                self.snake_controller.click_window_center()
                        except Exception as e:
                            self.logger.log(f"蛇控制错误: {str(e)}")

                        # 更新 UI 上的状态标签
                        self.game_state_label.setText(f"当前状态: {game_state}")

                        draw_start = time.time()
                        # 如果开启了分析结果显示，使用 map_drawer 绘制地图
                        if self.btn_show_analysis.isChecked():
                            screen_cv = self.map_drawer.draw_map(board_state, path, screen_cv)

                        # 将处理后的图像转换回RGB格式
                        screen_cv = cv2.cvtColor(screen_cv, cv2.COLOR_BGR2RGB)
                        # 将OpenCV图像转换回PIL格式
                        screen = Image.fromarray(screen_cv)

                        draw_time = time.time() - draw_start

                        # 更新性能计时显示栏
                        self.convert_time_badge.setText(f"图像转换: {convert_time:.3f}s")
                        self.analyze_time_badge.setText(f"棋盘分析: {analyze_time:.3f}s")
                        self.path_calc_time_badge.setText(f"路径计算: {path_calc_time:.3f}s")
                        self.draw_time_badge.setText(f"绘制: {draw_time:.3f}s")
                        
                        # 更新蛇的方向显示
                        if path and len(path) > 1:
                            current_pos = path[0]
                            next_pos = path[1]
                            dx = next_pos[0] - current_pos[0]
                            dy = next_pos[1] - current_pos[1]
                            direction = ""
                            if dx > 0:
                                direction = "右"
                            elif dx < 0:
                                direction = "左"
                            elif dy > 0:
                                direction = "下"
                            elif dy < 0:
                                direction = "上"
                            self.snake_direction_badge.setText(f"方向: {direction}")
                        else:
                            self.snake_direction_badge.setText("方向: 无")

                    except Exception as e:
                        self.status_label.setText(f"运行错误: {str(e)}")
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
            
            # 添加演示定时器并立即执行
            demo_timer = QTimer(self)
            demo_timer.setSingleShot(True)
            demo_timer.timeout.connect(lambda: self.window().adjustSize())
            demo_timer.start(1000)  # 立即执行
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

    def onScreenToggled(self, isChecked: bool):
        """处理画面刷新开关的状态变化"""
        text = '开启画面刷新' if isChecked else '关闭画面刷新'
        self.btn_show_screen.setText(text)
        
    def onPathCalcFrequencyChanged(self, index: int):
        """处理路径计算频率变化"""
        if not hasattr(self, 'path_calculator'):
            return
            
        # 根据选择设置不同的路径计算间隔
        if index == 0:  # 高频率 (每帧)
            self.path_calculator.set_path_calc_interval(0.01)
            self.logger.log("路径计算频率设置为: 高 (每帧)")
        elif index == 1:  # 中频率
            self.path_calculator.set_path_calc_interval(0.1)
            self.logger.log("路径计算频率设置为: 中 (100ms)")
        elif index == 2:  # 低频率
            self.path_calculator.set_path_calc_interval(0.2)
            self.logger.log("路径计算频率设置为: 低 (200ms)")

    def select_window(self):
        """选择要捕获的窗口"""
        # 自动查找标题为'绝区零'的窗口
        hwnd = win32gui.FindWindow(None, '绝区零')
        if hwnd:
            self.selected_hwnd = hwnd
            self.status_label.setText("状态: 已找到'绝区零'窗口")
        else:
            self.status_label.setText("状态: 未找到'绝区零'窗口")