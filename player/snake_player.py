import cv2
import numpy as np
import time
import win32gui
from PyQt5.QtCore import QObject, pyqtSignal
from PIL import ImageGrab
from analyzer.board_analyzer import BoardAnalyzer
from model.snake_board import Board
from analyzer.path_finder import PathFinder
from controller.snake_controller import SnakeController
import mss
import mss.tools

class SnakePlayer(QObject):
    # 定义信号：传递游戏状态、棋盘、性能统计数据、当前控制方向、路径等
    board_updated = pyqtSignal(object, object, float, float, float, float,float, str, object)
    
    def __init__(self, logger=None):
        """
        初始化SnakePlayer核心对象

        输入参数:
            logger: 可选的日志记录器

        初始化内容:
            - 创建棋盘分析器、路径搜索器、控制执行器
            - 初始化状态参数、性能参数
            - 设置缓存区域
        """
        super().__init__()
        self.logger = logger
        # 初始化各个组件，确保传入logger
        self.board_analyzer = BoardAnalyzer(logger)
        self.path_finder = PathFinder(25, 29, logger)
        self.snake_controller = SnakeController(logger)
        self.head_next_pos = None   # 蛇头下一步位置，预留未使用

        # 性能统计（单位秒）
        self.convert_time = 0
        self.analyze_time = 0
        self.path_calc_time = 0
        self.draw_time = 0
        self.test_time = 0

        self.current_direction = None       # 当前的移动方向
        self.current_path = []              # 蛇的计划路径
        self.last_click_time = 0            # 上次点击窗口时间
        self.last_direction_time = 0        # 上次方向控制时间
        self.last_process_time = None       # 记录上一次调用时间
        self.time_since_last_frame = None   # 循环间隔

        # 缓存截图，便于调试
        self.cached_screen = None
        self.last_cache_update_time = 0  # 新增，缓存的上次更新时间戳
        self.cache_images = [None] * 5  # 新增缓存池，最多5张图
        self.cache_index = 0            # 新增当前写入索引

    def capture_screen(self, hwnd):
        """
        截取指定窗口句柄的游戏画面，裁剪出棋盘区域

        输入参数:
            hwnd - 游戏窗口的窗口句柄

        输出:
            screen_cv - 裁切后的游戏棋盘区域OpenCV BGR图像
                        捕获失败返回None

        处理流程:
            - 获取窗口客户区矩形（相对坐标）
            - 转换为屏幕绝对坐标
            - 使用PIL抓取整个窗口内容
            - 调整分辨率至1920x1080，固定缩放方便定位
            - 裁剪出棋盘所在区域
            - 转换为OpenCV格式
        """
        try:
            if hwnd is None or hwnd == 0:
                if self.logger:
                    self.logger.log("未检测到有效的游戏窗口句柄，蛇玩家未启动")
                return None

            client_left, client_top, client_right, client_bottom = win32gui.GetClientRect(hwnd)
            left, top = win32gui.ClientToScreen(hwnd, (client_left, client_top))
            right, bottom = win32gui.ClientToScreen(hwnd, (client_right, client_bottom))

            width = right - left
            height = bottom - top

            # 检查宽高比，必须接近16:9，否则报错
            if abs(width / height - 16 / 9) > 0.01:
                if self.logger:
                    self.logger.log(f"窗口分辨率比例非16:9，当前为{width}x{height}，捕获中止")
                return None

            # 按照窗口分辨率动态计算棋盘区域百分比
            # 实测你之前截图用的区域为 1920x1080 中的485,203 到 1434,1028
            # 根据这两点换算相对比例
            x_ratio1 = 485 / 1920
            y_ratio1 = 203 / 1080
            x_ratio2 = 1434 / 1920
            y_ratio2 = 1028 / 1080

            crop_left = int(left + width * x_ratio1)
            crop_top = int(top + height * y_ratio1)
            crop_right = int(left + width * x_ratio2)
            crop_bottom = int(top + height * y_ratio2)
            bbox = {
                "left": crop_left,
                "top": crop_top,
                "width": crop_right - crop_left,
                "height": crop_bottom - crop_top
            }
            
            # 直接截图棋盘区域
            with mss.mss() as sct:
                grab_img = sct.grab(bbox)
                img = np.array(grab_img)
                current_time = time.time()
                if current_time - self.last_cache_update_time >= 1.0:
                    # 写入循环缓存
                    self.cache_images[self.cache_index] = img.copy()
                    self.cache_index = (self.cache_index + 1) % 5

                    self.last_cache_update_time = current_time
                screen_cv = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            current_time = time.time()

            return screen_cv

        except Exception as e:
            if self.logger:
                self.logger.log(f"画面捕获错误: {str(e)}")
            return None

    def update_and_evaluate_path(self, board):
        """
        对当前路径进行更新和校验，剔除不合理节点

        输入参数:
            board - 当前棋盘状态对象

        输出:
            self.current_path - 过滤和校验后的路径列表

        处理流程:
            - 如果路径为空，直接返回
            - 过滤掉路径中为蛇身的节点
            - 检查剩余路径节点是否全为允许类型，否则清空路径
        """
        if not self.current_path:
            return self.current_path

        # 识别蛇头坐标
        head_pos = None
        if "own_head" in board.special_cells and board.special_cells["own_head"]:
            head_cell = board.special_cells["own_head"][0]
            head_pos = (head_cell.col, head_cell.row)

        # 蛇头不为None则重置路径起点
        if head_pos is not None:
            if not self.current_path or self.current_path[0] != head_pos:
                self.current_path = [head_pos] + self.current_path

        # 过滤掉身体节点，从第2个节点开始
        filtered_path = []
        if self.current_path:
            filtered_path.append(self.current_path[0])  # 保留蛇头

        for pos in self.current_path[1:]:
            cell_type = board.cells[pos[1]][pos[0]].cell_type
            if cell_type == "own_body":
                continue
            filtered_path.append(pos)

        self.current_path = filtered_path

        # **剔除路径中连续重复的点**
        deduped_path = []
        for pos in self.current_path:
            if not deduped_path or pos != deduped_path[-1]:
                deduped_path.append(pos)

        self.current_path = deduped_path

        # 新增：剪除回头路（遇到重复坐标，就截断到第一次出现该点）
        visited_pos = dict()
        for idx, pos in enumerate(self.current_path):
            if pos in visited_pos:
                # 截断路径，只保留头到该重复点第一次出现为止
                self.current_path = self.current_path[:visited_pos[pos]+1]
                break
            visited_pos[pos] = idx

        # 校验剩余路径是否安全（蛇头之外）
        for pos in self.current_path[1:]:
            cell_type = board.cells[pos[1]][pos[0]].cell_type
            if cell_type not in ["empty", "score_boost", "own_head", "unknow"]:
                self.current_path = []
                break

        # 校验蛇头与下一节点的连续性
        if head_pos is not None and len(self.current_path) >= 2:
            next_pos = self.current_path[1]
            dx = abs(next_pos[0] - head_pos[0])
            dy = abs(next_pos[1] - head_pos[1])
            if dx + dy != 1:
                self.current_path = []

        return self.current_path

    def process_frame(self, screen_cv, hwnd):
        """
        处理一帧捕获的游戏画面，判断状态，计算路径，控制操作

        输入参数:
            screen_cv - 当前游戏棋盘区域的OpenCV格式图像
            hwnd - 游戏窗口句柄

        输出:
            三元组 (game_state, board, self.current_path)
                game_state: 当前识别到的游戏状态
                board: 棋盘状态对象
                path: 当前规划路径（列表）
        """
        # 检查必要Qt对象是否存在
        if not hasattr(self, 'board_updated') or not hasattr(self, 'logger'):
            return None, None, []

        # 更新处理时间戳及循环间隔
        current_time = time.time()
        self.time_since_last_frame = 0 if self.last_process_time is None else current_time - self.last_process_time
        self.last_process_time = current_time

        try:
            analyze_start = time.time()
            try:
                board = Board(25, 29, image=screen_cv, image_format='BGR')
                self.board = self.board_analyzer.analyze_board(board, last_key_direction=self.current_direction)

                if self.board_analyzer.is_gameover:
                    game_state = "game_over"
                elif self.board_analyzer.is_running:
                    game_state = "running"
                else:
                    game_state = "waiting"

            except Exception as e:
                self.logger.log(f"图像分析错误: {str(e)}")
                game_state, board = None, None

            self.analyze_time = time.time() - analyze_start

            # 如果识别没成功就返回当前状态
            if not board:
                return game_state, None, []

            # 更新路径
            self.current_path = self.update_and_evaluate_path(board)
            
            # 控制蛇移动：必须路径存在且节点≥2才行
            self.control_snake_by_path(hwnd)

            # 计算新路径（只在运行中且路径为空或只有1个点时）
            
            if game_state == "running":
                try:
                    if not self.current_path or len(self.current_path) <= 1:
                        # 寻路
                        path_calc_start = time.time()
                        self.current_path = self.path_finder.find_path_in_order(board)
                        self.path_calc_time = time.time() - path_calc_start
                        self.control_snake_by_path(hwnd)

                except Exception as e:
                    self.logger.log(f"路径计算或控制错误: {str(e)}")
                    self.current_path = []
            
            if game_state == "game_over":
                current_time = time.time()
                if current_time - self.last_click_time >= 1.0:
                    self.snake_controller.set_game_window(hwnd)
                    self.snake_controller.click_window_center()
                    self.last_click_time = current_time

            self.board_updated.emit(
                game_state, board,
                self.convert_time, self.analyze_time, self.path_calc_time, self.draw_time,
                self.time_since_last_frame, self.current_direction, self.current_path
            )
            return game_state, board, self.current_path

        except Exception as e:
            if self.logger:
                self.logger.log(f"处理帧总异常: {str(e)}")
            return None, None, []

    def control_snake_by_path(self, hwnd):
        """
        根据当前路径和句柄控制蛇的移动方向
        """
        if self.current_path and len(self.current_path) >= 2:
            self.snake_controller.set_game_window(hwnd)
            current_pos = self.current_path[0]
            next_pos = self.current_path[1]
            dx = next_pos[0] - current_pos[0]
            dy = next_pos[1] - current_pos[1]

            next_direction = None
            if dx > 0:
                next_direction = "right"
            elif dx < 0:
                next_direction = "left"
            elif dy > 0:
                next_direction = "down"
            elif dy < 0:
                next_direction = "up"

            if next_direction is not None:
                self.current_direction = next_direction
                self.snake_controller.control_snake(next_direction)
