import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QLabel
from PyQt5.QtWidgets import QSizePolicy
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    TextBrowser,
    SwitchButton,
    InfoBadge,
    PrimaryPushButton,
)
import os
from log.log import SnakeLogger
from drawer.map_drawer import MapDrawer
from log.debug_helper import DebugHelper  # 新增导入


class SnakeGameCard(QFrame):
    def __init__(self, snake_player_thread=None, parent=None):
        super().__init__(parent)
        self.setObjectName("SnakeGameCard")
        self.snake_player_thread = snake_player_thread

        # 主布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 20)  # 上下边距从30改为20
        layout.setSpacing(10)  # 从20改为10，减小各部分之间的间距
        layout.setAlignment(Qt.AlignCenter)

        # 顶部显示区布局
        top_layout = QVBoxLayout()
        top_layout.setSpacing(0)

        # 游戏截图区域
        self.screen_label = QLabel()
        self.screen_label.setAlignment(Qt.AlignCenter)
        self.screen_label.setFixedSize(500, 500)
        self.screen_label.setStyleSheet(
            """
            QLabel {
                border-radius: 8px;
                background-color: black;
                min-width: 500px;
                min-height: 500px;
            }
        """
        )

        # 按钮：显示分析开关
        self.btn_show_analysis = SwitchButton(parent=self)
        self.btn_show_analysis.setChecked(True)
        self.btn_show_analysis.checkedChanged.connect(self.onAnalysisToggled)

        # 按钮：蛇玩家自动控制开关
        self.btn_snake_player = SwitchButton(parent=self)
        self.btn_snake_player.setChecked(False)
        self.btn_snake_player.setText("启用蛇玩家")
        self.btn_snake_player.checkedChanged.connect(self.onSnakePlayerToggled)

        # 状态文本
        self.status_label = CaptionLabel("状态: 准备就绪")
        self.status_label.setAlignment(Qt.AlignCenter)

        # 日志文本框
        self.log_text = TextBrowser()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedWidth(500)
        self.log_text.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)

        # 游戏状态标签
        self.game_state_label = BodyLabel("当前状态: 初始化")
        self.game_state_label.setAlignment(Qt.AlignCenter)

        # ===== 只有QFluentWidgets状态栏标签，去除字体大小调整 =====
        self.status_label = BodyLabel("蛇玩家: 准备就绪")
        self.status_label.setAlignment(Qt.AlignCenter)

        self.game_state_label = BodyLabel("棋盘: 初始化")
        self.game_state_label.setAlignment(Qt.AlignCenter)

        self.status_bar_layout = QHBoxLayout()
        self.status_bar_layout.setSpacing(30)
        self.status_bar_layout.setAlignment(Qt.AlignCenter)
        self.status_bar_layout.addWidget(self.status_label)
        self.status_bar_layout.addWidget(self.game_state_label)

        self.status_bar_container = QFrame()
        self.status_bar_container.setObjectName("StatusBarFrame")
        self.status_bar_container.setLayout(self.status_bar_layout)
        self.status_bar_container.setStyleSheet(
            """
            #StatusBarFrame {
                background-color: transparent;
                border-top: 1px solid #d0d0d0;
                padding: 3px 0;  /* 从6px改为3px，减小上下内边距 */
            }
        """
        )
        # ===== END =====

        # 性能信息展示
        self.performance_layout = QHBoxLayout()
        self.convert_time_badge = InfoBadge.custom("转换: 0.000s", "#0066cc", "#ffffff")
        self.analyze_time_badge = InfoBadge.custom("分析: 0.000s", "#009933", "#ffffff")
        self.path_time_badge = InfoBadge.custom("寻路: 0.000s", "#cc6600", "#ffffff")
        self.draw_time_badge = InfoBadge.custom("绘制: 0.000s", "#6600cc", "#ffffff")
        self.player_time_badge = InfoBadge.custom("循环: 0.000s", "#6699cc", "#ffffff")
        self.snake_direction_badge = InfoBadge.custom("方向: 无", "#cc3300", "#ffffff")

        self.snake_pos_badge = InfoBadge.custom(
            "坐标: 无", "#333399", "#ffffff"
        )  # 新增蛇头和下一步坐标标签

        # self.performance_layout.addWidget(self.convert_time_badge)
        self.performance_layout.addWidget(self.analyze_time_badge)
        self.performance_layout.addWidget(self.path_time_badge)
        # self.performance_layout.addWidget(self.draw_time_badge)
        self.performance_layout.addWidget(self.player_time_badge)
        # self.performance_layout.addWidget(self.snake_direction_badge)
        self.performance_layout.addWidget(self.snake_pos_badge)  # 新增加入性能信息栏
        self.performance_layout.setSpacing(10)
        self.performance_layout.setAlignment(Qt.AlignCenter)

        self.performance_container = QFrame()
        self.performance_container.setLayout(self.performance_layout)

        # 顶部加入截图
        top_layout.addWidget(self.screen_label)

        # 中部按钮区
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        button_layout.addWidget(self.btn_show_analysis)
        button_layout.addWidget(self.btn_snake_player)

        self.btn_screenshot = PrimaryPushButton("保存截图")
        self.btn_screenshot.clicked.connect(self._start_batch_screenshot)
        button_layout.addWidget(self.btn_screenshot)

        # 底部区
        bottom_layout = QVBoxLayout()
        bottom_layout.setSpacing(3)  # 从5改为3
        bottom_layout.addWidget(self.log_text)
        bottom_layout.addWidget(self.performance_container)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        layout.addLayout(top_layout)
        layout.addLayout(button_layout)
        layout.addLayout(bottom_layout)
        layout.addWidget(self.status_bar_container)  # 添加美化后状态栏作为底部

        # 初始化logger和地图绘制器
        self.logger = SnakeLogger(self.log_text)
        self.map_drawer = MapDrawer()

        # 截图批量相关
        self.batch_screenshot_timer = QTimer(self)
        self.batch_screenshot_timer.setInterval(1000)
        self.batch_screenshot_timer.timeout.connect(self._batch_screenshot_tick)
        self.batch_screenshot_count = 0

        # 缓存缩放后的pixmap，避免频繁缩放
        self.scaled_pixmap = None
        self.last_draw_img_shape = None
        self.last_label_size = self.screen_label.size()

        # 调试图片保存目录
        self.debug_images_dir = os.path.join(
            os.path.dirname(__file__), ".debug", "images"
        )
        os.makedirs(self.debug_images_dir, exist_ok=True)

    def _start_batch_screenshot(self):
        """
        一次性保存缓存的5张历史截图
        """
        if not (
            self.snake_player_thread
            and hasattr(self.snake_player_thread.snake_player, "cache_images")
        ):
            self.logger.warning("无缓存数组，无法保存")
            return

        cached_images = self.snake_player_thread.snake_player.cache_images
        saved_count = 0
        for idx, img in enumerate(cached_images):
            if img is not None:
                try:
                    # 复制OpenCV图像为BGR格式保存
                    bgr_img = img
                    saved_path = DebugHelper.save_image(
                        bgr_img, prefix=f"snake_cache_{idx}"
                    )
                    if saved_path:
                        filename = os.path.basename(saved_path)
                        self.logger.info(f"已保存缓存第{idx+1}张: {filename}")
                        saved_count += 1
                    else:
                        self.logger.error(f"缓存第{idx+1}张保存失败：返回空路径")
                except Exception as e:
                    self.logger.error(f"缓存第{idx+1}张保存出错: {str(e)}")
            else:
                self.logger.warning(f"缓存第{idx+1}张为空，跳过")

        self.logger.info(f"缓存截图保存完成，共保存 {saved_count}/5 张")

    def _batch_screenshot_tick(self):
        """
        定时器槽函数，保存截图
        """
        if (
            self.snake_player_thread
            and self.snake_player_thread.snake_player.cached_screen is not None
        ):
            try:
                rgb_img = np.array(self.snake_player_thread.snake_player.cached_screen)
                bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                saved_path = DebugHelper.save_image(bgr_img, prefix="snake")
                if saved_path:
                    filename = os.path.basename(saved_path)
                    self.logger.info(f"截图已保存: {filename}")
                else:
                    self.logger.error("保存截图失败：返回路径为空")
            except Exception as e:
                self.logger.error(f"保存截图失败: {str(e)}")
        else:
            self.logger.warning("没有可用的截图")

        self.batch_screenshot_count -= 1
        if self.batch_screenshot_count <= 0:
            self.batch_screenshot_timer.stop()
            self.logger.info("连续3张截图完成")

    @pyqtSlot(object, object, float, float, float, float, float, str, object)
    def on_board_updated(
        self,
        game_status,
        board,
        image_convert_time,
        analysis_time,
        path_find_time,
        render_time,
        flash_time,
        snake_direction,
        path,
    ):
        if game_status:
            self.game_state_label.setText(f"棋盘: {game_status}")

        self.convert_time_badge.setText(f"转换: {image_convert_time:.3f}s")
        self.analyze_time_badge.setText(f"分析: {analysis_time:.3f}s")
        self.path_time_badge.setText(f"寻路: {path_find_time:.3f}s")
        self.draw_time_badge.setText(f"绘制: {render_time:.3f}s")
        self.player_time_badge.setText(f"循环: {flash_time:.3f}s")
        self.snake_direction_badge.setText(
            f"方向: {snake_direction if snake_direction else '无'}"
        )

        # 新增：更新蛇头和路径前两个节点的坐标显示
        real_head_text = "无"
        if (
            board
            and "own_head" in board.special_cells
            and board.special_cells["own_head"]
        ):
            cell = board.special_cells["own_head"][0]
            real_head_text = f"({cell.col},{cell.row})"

        path_text = ""
        if path:
            if len(path) >= 2:
                path_text = f"{path[0]}->{path[1]}"
            elif len(path) == 1:
                path_text = f"{path[0]}"
            else:
                path_text = "无"
        else:
            path_text = "无"

        self.snake_pos_badge.setText(f"蛇头:{real_head_text} 路径:{path_text}")

        if board:
            try:
                draw_img = self.map_drawer.draw_map(board, path)

                height, width = draw_img.shape[:2]
                bytes_per_line = 3 * width
                rgb_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
                q_img = QImage(
                    rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888
                )

                pixmap = QPixmap.fromImage(q_img)
                label_size = self.screen_label.size()

                scaled_pixmap = pixmap.scaled(
                    label_size.width(),
                    label_size.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self.screen_label.setPixmap(scaled_pixmap)

            except Exception as e:
                self.logger.log(f"绘制错误: {str(e)}")

    def onAnalysisToggled(self, isChecked):
        """
        分析结果显示开关切换回调

        输入参数：
            isChecked (bool) - 是否启用分析结果显示

        输出：无

        处理流程：
            - 修改按钮文字
        """
        text = "开启分析" if isChecked else "关闭分析"
        self.btn_show_analysis.setText(text)

    def onSnakePlayerToggled(self, isChecked):
        """
        蛇玩家自动控制开关处理函数

        输入参数：
            isChecked (bool) - 当前开关状态

        输出：
            无

        处理流程：
            - 查找游戏窗口
            - 初始化蛇玩家线程
            - 根据开关启停线程
        """
        text = "停用蛇玩家" if isChecked else "启用蛇玩家"
        self.btn_snake_player.setText(text)
        self.logger.info("切换了蛇玩家开关")

        if self.snake_player_thread:
            if isChecked:
                import win32gui

                hwnd = win32gui.FindWindow(None, "绝区零")
                if hwnd:
                    self.snake_player_thread.set_hwnd(hwnd)
                    self.snake_player_thread.snake_player.logger = self.logger
                    self.snake_player_thread.snake_player.board_analyzer.logger = (
                        self.logger
                    )
                    self.snake_player_thread.snake_player.path_finder.logger = (
                        self.logger
                    )
                    self.snake_player_thread.snake_player.snake_controller.logger = (
                        self.logger
                    )
                    if not self.snake_player_thread.isRunning():
                        self.snake_player_thread.start()
                    self.logger.info("蛇玩家线程已启动")
                else:
                    self.logger.warning("未找到游戏窗口，无法启动蛇玩家线程")
                    self.btn_snake_player.setChecked(False)
            else:
                if self.snake_player_thread.isRunning():
                    self.snake_player_thread.stop()
                    self.snake_player_thread.wait()  # 等待线程完全停止
                self.logger.info("蛇玩家线程已停止")
        else:
            self.logger.warning("蛇玩家线程对象不存在，无法执行操作")
