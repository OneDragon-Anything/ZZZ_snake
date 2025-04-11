import cv2
import numpy as np
import time
import win32gui
from PyQt5.QtCore import QObject, pyqtSignal, QThread, pyqtSlot, QMutex
from PIL import ImageGrab
from analyzer.board_analyzer import BoardAnalyzer
from model.snake_board import Board
from analyzer.path_finder import PathFinder
from controller.snake_controller import SnakeController
import mss
import mss.tools
import sys
import gc  # 垃圾回收
from log.debug_helper import DebugHelper

class PathWorker(QObject):
    """负责异步计算路径的线程工作对象"""

    finished = pyqtSignal(object)
    calculate = pyqtSignal(object)

    def __init__(self, path_finder: PathFinder):
        """
        :param path_finder: 路径查找器实例
        """
        super().__init__()
        self.path_finder = path_finder
        self.is_calculating = False
        self.mutex = QMutex()
        self.last_fail_screenshot_time = 0

    @pyqtSlot(object)
    def run(self, board: Board):
        """
        工作线程调用此函数，计算路径。
        :param board: 当前棋盘对象
        """
        if self.mutex.tryLock():
            try:
                if not self.is_calculating:
                    self.is_calculating = True
                    result = self.path_finder.find_path_in_order(board)

                    now = time.time()
                    # 路径为空超5秒截图
                    if (result is None or not result) and (now - self.last_fail_screenshot_time > 5):
                        try:
                            file_path = DebugHelper.save_image(board.bgr_image, prefix="no_path_thread")
                            self.last_fail_screenshot_time = now
                            if self.path_finder.logger:
                                self.path_finder.logger.log(f"[线程] 路径为空已截图: {file_path}")
                        except Exception as e:
                            if self.path_finder.logger:
                                self.path_finder.logger.log(f"[线程] 截图异常: {e}")
                    self.finished.emit(result)
                    self.is_calculating = False
            finally:
                self.mutex.unlock()

class SnakePlayer(QObject):
    """负责整体蛇控制、界面更新、路径计算的控制器"""
    board_updated = pyqtSignal(object, object, float, float, float, float, float, str, object)

    def __init__(self, logger=None):
        """
        :param logger: 日志记录器（可为空）
        """
        super().__init__()
        self.logger = logger
        self.board_analyzer = BoardAnalyzer(logger)
        self.path_finder = PathFinder(25, 29, logger)
        self.snake_controller = SnakeController(logger)
        self.head_next_pos = None
        self.board = None

        self.convert_time = 0
        self.analyze_time = 0
        self.path_calc_time = 0
        self.draw_time = 0
        self.test_time = 0
        self.last_time_control_snake_head = None

        self.current_direction = None
        self.current_path = []
        self.last_click_time = 0
        self.last_direction_time = 0
        self.last_process_time = None
        self.time_since_last_frame = None

        self.cached_screen = None
        self.last_cache_update_time = 0
        self.cache_images = [None] * 10
        self.cache_index = 0
        self.last_hwnd = 0
        self.last_cache_cleanup_time = time.time()
        self.last_path_calc_time = 0
        self.path_calc_interval = 0.2

        self.path_thread = QThread()
        self.path_worker = PathWorker(self.path_finder)
        self.path_worker.moveToThread(self.path_thread)
        self.path_thread.start()
        self.path_worker.finished.connect(self.on_path_finished)
        self.path_worker.calculate.connect(self.path_worker.run)

    def __del__(self):
        """析构，释放缓存、线程"""
        try:
            self.path_thread.quit()
            self.path_thread.wait()
            for i in range(len(self.cache_images)):
                self.cache_images[i] = None
            gc.collect()
        except:
            pass

    def capture_screen(self, hwnd) -> 'np.ndarray|None':
        """
        截取游戏窗口内容
        :param hwnd: 游戏窗口句柄
        :return: 图像BGR数组 (None失败)
        """
        try:
            if not hwnd:
                if self.logger:
                    self.logger.log("未检测到有效的游戏窗口句柄，蛇玩家未启动")
                return None

            cl, ct, cr, cb = win32gui.GetClientRect(hwnd)
            left, top = win32gui.ClientToScreen(hwnd, (cl, ct))
            right, bottom = win32gui.ClientToScreen(hwnd, (cr, cb))

            width = right - left
            height = bottom - top

            if abs(width / height - 16 / 9) > 0.01:
                if self.logger:
                    self.logger.log(f"窗口比例非16:9，当前{width}x{height}，放弃截图")
                return None

            x1 = int(left + width * 485 / 1920)
            y1 = int(top + height * 203 / 1080)
            x2 = int(left + width * 1434 / 1920)
            y2 = int(top + height * 1028 / 1080)

            bbox = {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1}

            with mss.mss() as sct:
                sct_img = sct.grab(bbox)
                img = np.array(sct_img)

                current_time = time.time()
                if current_time - self.last_cache_update_time >= 1.0:
                    if self.cache_images[self.cache_index] is not None:
                        self.cache_images[self.cache_index] = None

                    self.cache_images[self.cache_index] = img.copy()
                    self.cache_index = (self.cache_index + 1) % 10
                    self.last_cache_update_time = current_time

                    if current_time - self.last_cache_cleanup_time >= 30.0:
                        for i in range(len(self.cache_images)):
                            if i != self.cache_index and i != (self.cache_index - 1) % 10:
                                self.cache_images[i] = None
                        self.last_cache_cleanup_time = current_time
                        gc.collect()

                screen_cv = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            return screen_cv

        except Exception as e:
            if self.logger:
                self.logger.log(f"画面捕获错误: {e}")
            return None

    def update_and_evaluate_path(self, board: Board) -> list:
        """
        根据当前路径和敌人位置，对路径做二次评估
        :param board: 当前棋盘
        :return: 评估后的路径（或空）
        """
        if not self.current_path:
            return self.current_path

        enemy_heads = [
            (c.col, c.row) for c in board.special_cells.get('enemy_head', [])
        ]

        for step_index, pos in enumerate(self.current_path[1:6], start=1):
            cell_type = board.cells[pos[1]][pos[0]].cell_type
            if cell_type not in ["empty", "score_boost", "own_head", "unknow"]:
                if self.logger:
                    self.logger.debug(f"[路径评估] 清空路径：遇到障碍物，位置({pos[0]},{pos[1]})，类型={cell_type}")
                self.current_path = []
                break
            for (ex, ey) in enemy_heads:
                enemy_dist = abs(pos[0] - ex) + abs(pos[1] - ey)
                # 只有当敌人距离路径点1格内时才清空路径
                if enemy_dist <= 1:  # 修改为1格距离判断
                    if self.logger:
                        self.logger.debug(f"[路径评估] 清空路径：敌人蛇头({ex},{ey})距离路径点({pos[0]},{pos[1]})仅{enemy_dist}格")
                    self.current_path = []
                    return self.current_path

        return self.current_path

    def process_frame(self, screen_cv: 'np.ndarray', hwnd) -> tuple:
        """
        完成图像识别、路径计算、控制蛇全过程
        :param screen_cv: 当前截屏
        :param hwnd: 游戏窗口句柄
        :return: (游戏状态, 识别棋盘对象, 当前路径)
        """
        self.last_hwnd = hwnd
        if not hasattr(self, 'board_updated') or not hasattr(self, 'logger'):
            return None, None, []

        now = time.time()
        self.time_since_last_frame = 0 if self.last_process_time is None else now - self.last_process_time
        self.last_process_time = now

        try:
            analyze_start = time.time()
            try:
                if self.board is None:
                    self.board = Board(25, 29, image=screen_cv, image_format='BGR')
                else:
                    self.board.set_hsv_image(screen_cv, image_format='BGR')

                self.board = self.board_analyzer.analyze_board(
                    self.board, last_key_direction=self.current_direction
                )

                # 预测蛇头位置，如果看到了就无需预测
                self.predict_or_update_head(self.board)

                if self.board_analyzer.is_gameover:
                    game_state = "game_over"
                elif self.board_analyzer.is_running:
                    game_state = "running"
                else:
                    game_state = "waiting"

            except Exception as e:
                self.logger.log(f"图像分析错误: {e}")
                game_state = None

            self.analyze_time = time.time() - analyze_start

            if self.board is None or not hasattr(self.board, 'cells') or not isinstance(self.board.cells, list) or not self.board.cells:
                self.logger.log("analyze_board 返回棋盘非法")
                return game_state, None, []

            if len(self.board.cells) < 25 or len(self.board.cells[0]) < 29:
                self.logger.log(f"棋盘尺寸异常: {len(self.board.cells)}×{len(self.board.cells[0])}")
                return game_state, None, []

            try:
                self.control_snake_by_path(hwnd)
            except Exception as e:
                self.logger.log(f"控制蛇异常: {e}")

            try:
                self.current_path = self.update_and_evaluate_path(self.board)
            except Exception as e:
                self.logger.log(f"路径评估异常: {e}")
                self.current_path = []

            if game_state == "running":
                try:
                    current_time = time.time()
                    if (not self.current_path) or (current_time - self.last_path_calc_time >= 1.0):
                        calc_start = time.time()
                        self.path_worker.calculate.emit(self.board)
                        self.path_calc_time = time.time() - calc_start
                        self.last_path_calc_time = current_time
                except Exception as e:
                    self.logger.log(f"异步寻路触发异常: {e}")
                    self.current_path = []

            if game_state == "game_over":
                if time.time() - self.last_click_time >= 1.0:
                    try:
                        self.snake_controller.set_game_window(hwnd)
                        self.snake_controller.click_window_center()
                        self.last_click_time = time.time()
                    except Exception as e:
                        self.logger.log(f"自动点击复活异常: {e}")

            try:
                self.board_updated.emit(
                    game_state, self.board,
                    self.convert_time, self.analyze_time, self.path_calc_time, self.draw_time,
                    self.time_since_last_frame, self.current_direction, self.current_path
                )
            except Exception as e:
                self.logger.log(f"信号emit异常: {e}")

            return game_state, self.board, self.current_path

        except Exception as e:
            self.logger.log(f"处理帧异常: {e}")
            return None, None, []

    @pyqtSlot(object)
    def on_path_finished(self, path_result: list):
        """
        路径计算完成后回调
        :param path_result: 算法算出的路径
        """
        self.current_path = path_result
        try:
            self.control_snake_by_path(self.last_hwnd)
        except:
            pass

    def control_snake_by_path(self, hwnd):
        """
        根据当前路径控制蛇移动
        :param hwnd: 游戏窗口句柄
        """
        if not self.current_path:
            head_cell = self.board.special_cells.get("own_head", [None])[0]
            if not head_cell:
                return
            real_head = (head_cell.col, head_cell.row)
            self.try_emergency_avoidance(real_head, hwnd)
            return

        current_path = self.current_path.copy()

        if len(current_path) < 2:  # 至少需要两个点才能确定方向
            return

        head_cell = self.board.special_cells.get("own_head", [None])[0]
        if not head_cell:
            return

        real_head = (head_cell.col, head_cell.row)
        target = current_path[0]

        # 计算与当前目标点的距离
        dx = abs(real_head[0] - target[0])
        dy = abs(real_head[1] - target[1])
        manhattan_dist = dx + dy

        # 如果已经到达当前目标点，则选择下一个目标点
        if manhattan_dist == 0 and len(current_path) > 1:
            target = current_path[1]
            # 弹出已到达的点
            self.current_path.pop(0)

            # 重新计算与新目标的距离
            dx = abs(real_head[0] - target[0])
            dy = abs(real_head[1] - target[1])

        is_adjacent = (dx == 1 and dy == 0) or (dx == 0 and dy == 1)
        if not is_adjacent:
            if self.logger:
                self.logger.debug(f"[路径控制] 清空路径：目标点({target[0]},{target[1]})与蛇头({real_head[0]},{real_head[1]})不相邻(dx={dx}, dy={dy})")
            self.current_path = []
            return

        # 计算移动方向
        dx, dy = target[0]-real_head[0], target[1]-real_head[1]
        direction = None
        if dx == 1:
            direction = "right"
        elif dx == -1:
            direction = "left"
        elif dy == 1:
            direction = "down"
        elif dy == -1:
            direction = "up"

        if direction:
            self.snake_move(hwnd, direction, reason="路径跟随：前往当前最近的目标点")
        else:
            if self.logger:
                self.logger.debug("[路径控制] 无法确定移动方向")
        return

    def try_emergency_avoidance(self, real_head: tuple, hwnd):
        """
        无路径时尝试往安全格移动避免死亡
        :param real_head: 蛇头xy坐标
        :param hwnd: 游戏窗口句柄
        """
        x, y = real_head
        neighbors = [
            (x+1, y, "right"),
            (x-1, y, "left"),
            (x, y+1, "down"),
            (x, y-1, "up"),
        ]

        for nx, ny, direction in neighbors:
            if not (0 <= nx < self.board.cols and 0 <= ny < self.board.rows):
                continue
            cell = self.board.cells[ny][nx]
            if cell is None:
                continue
            ctype = cell.cell_type
            if ctype not in ["empty", "score_boost", "own_head", "unknow"]:
                continue

            self.snake_move(hwnd, direction, reason="遇到危险：敌人接近，紧急避让")
            break

    def snake_move(self, hwnd, direction, reason=""):
        """
        记录当前蛇的最新方向键和时间
        :param direction: 'up' 'down' 'left' 'right'
        :param reason: 本次控制蛇移动的理由（调试用）
        """


        now = time.time()
        if hasattr(self, "last_snake_move_time") and now - self.last_snake_move_time < 0.02:
            return
        self.last_snake_move_time = now

        # 防御非法direction
        if not isinstance(direction, str) or direction.lower() not in ['up', 'down', 'left', 'right']:
            if self.logger:
                self.logger.log(f"[Debug] 无效direction参数: {direction} ({type(direction)})")
            else:
                print(f"[Debug] 无效direction参数: {direction} ({type(direction)})")
            return


        # if self.logger:
        #     msg = f"snake_move触发: 方向={direction}"
        #     if reason:
        #         msg += f"，原因={reason}"
        #     self.logger.debug(msg)

        self.snake_controller.set_game_window(hwnd)
        self.snake_controller.control_snake(direction)
        self.current_direction = direction
        self.last_direction_time = now

        # 简单起见，直接获取第一个蛇头
        if self.board and self.board.special_cells.get("own_head"):
            cell = self.board.special_cells["own_head"][0]
            self.last_time_control_snake_head = (cell.col, cell.row)
        else:
            self.last_time_control_snake_head = None

    def predict_or_update_head(self, board: Board):
        """
        检测棋盘是否有我方蛇头，如果没有则根据上次控制信息预测蛇头位置，并加入特殊格
        :param board: 当前棋盘
        """
        # 如果已经识别到了蛇头，则无需预测
        has_head = board.special_cells.get("own_head")
        if has_head and len(has_head) > 0:
            return

        # 条件不足，无法预测
        if not self.last_time_control_snake_head or not self.current_direction or not self.last_direction_time:
            return

        # 计算蛇头预测步数
        elapsed_time = time.time() - self.last_direction_time
        steps = int(elapsed_time / 0.2)

        x, y = self.last_time_control_snake_head

        dx, dy = 0, 0
        if self.current_direction == "up":
            dy = -1
        elif self.current_direction == "down":
            dy = 1
        elif self.current_direction == "left":
            dx = -1
        elif self.current_direction == "right":
            dx = 1

        px = x + steps * dx
        py = y + steps * dy

        # 限制在棋盘范围内
        if px < 0 or px >= board.cols or py < 0 or py >= board.rows:
            return

        # 将预测蛇头位置对应cell设为蛇头
        cell = board.cells[py][px]
        cell.cell_type = "own_head"

        # 确保字典有对应类型的列表（避免KeyError）
        if cell.cell_type not in board.special_cells:
            board.special_cells[cell.cell_type] = []

        # 先移除相同坐标的cell
        board.special_cells[cell.cell_type] = [
            c for c in board.special_cells[cell.cell_type]
            if not (c.row == cell.row and c.col == cell.col)
        ]

        # 添加新的cell
        board.special_cells[cell.cell_type].append(cell)
        return
