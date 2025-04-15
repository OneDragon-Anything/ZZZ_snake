import cv2
import numpy as np
import time
import win32gui
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import gc
from analyzer.board.board_analyzer import BoardAnalyzer
from model.snake_board import Board
from analyzer.path.path_finder import PathFinder
from controller.snake_controller import SnakeController
from log.debug_helper import DebugHelper


class SnakePlayer(QObject):
    """
    蛇玩家类：负责游戏控制、路径规划和界面更新
    主要功能：
    1. 游戏画面捕获和分析
    2. 路径规划和评估
    3. 蛇的移动控制
    4. 状态更新和信号发送
    """

    # 状态更新信号：游戏状态、棋盘、各阶段耗时、当前方向、路径
    board_updated = pyqtSignal(
        object, object, float, float, float, float, float, str, object
    )

    # 常量定义
    MOVE_INTERVAL = 0.02
    REVIVE_INTERVAL = 1.0
    BOARD_WIDTH = 29
    BOARD_HEIGHT = 25
    VALID_DIRECTIONS = {"up", "down", "left", "right", "stop"}
    SAFE_CELL_TYPES = {
        "empty",
        "score_boost",
        "own_head",
        "unknown",
        "own_tail",
        "own_body",
    }

    def __init__(self, logger=None):
        super().__init__()
        # 核心组件初始化
        self.logger = logger
        self.board_analyzer = BoardAnalyzer(logger)
        self.path_finder = PathFinder(self.BOARD_HEIGHT, self.BOARD_WIDTH, logger)
        self.snake_controller = SnakeController(logger)

        # 游戏状态相关
        self.board = None
        self.current_direction = None
        self.current_path = []
        self.head_next_pos = None
        
        # 蛇头位置相关
        self.observed_head_pos = None  # 实际观察到的蛇头位置
        self.predicted_head_pos = None  # 预测的蛇头位置
        self.predicted_body_positions = []  # 预测的蛇身位置列表(最多保持5格)

        # 时间控制相关
        self.last_cache_update_time = 0
        self.last_click_time = 0
        self.last_direction_time = 0
        self.last_process_time = None
        self.last_hwnd = 0
        self.last_snake_move_time = 0
        self.last_time_control_snake_head = None

        # 性能统计相关
        self.convert_time = 0
        self.analyze_time = 0
        self.path_calc_time = 0
        self.draw_time = 0
        self.time_since_last_frame = 0

        # 调试相关
        self.reason = None
        
        # 路径方向缓存
        self.cached_directions = []
        self.last_execute_time = 0

    def process_frame(self, screen_cv: "np.ndarray", hwnd) -> tuple:
        """处理当前帧"""
        self.last_hwnd = hwnd
        if not self._check_prerequisites():
            return None, None, []

        self._update_frame_time()

        try:
            game_state, board = self._analyze_frame(screen_cv)
            if not self._validate_board(board):
                return game_state, None, []

            self._evaluate_current_path()

            if game_state == "running":
                self._handle_running_state(hwnd)
            elif game_state == "game_over":
                if self.logger:
                    self.logger.debug("[状态] 游戏结束，准备重新开始")
                self._handle_game_over_state(hwnd)

            self._emit_board_update(game_state)
            return game_state, self.board, self.current_path

        except Exception as e:
            if self.logger:
                self.logger.error(f"[错误] 处理帧异常: {str(e)}")
            return None, None, []

    def _check_prerequisites(self) -> bool:
        """检查必要条件"""
        return hasattr(self, "board_updated") and hasattr(self, "logger")

    def _update_frame_time(self):
        """更新帧时间"""
        now = time.time()
        self.time_since_last_frame = (
            0 if self.last_process_time is None else now - self.last_process_time
        )
        self.last_process_time = now

    def _analyze_frame(self, screen_cv: "np.ndarray") -> tuple:
        """分析当前帧"""
        analyze_start = time.time()
        try:
            if self.board is None:
                if self.logger:
                    self.logger.debug("[分析] 初始化新棋盘")
                self.board = Board(
                    self.BOARD_HEIGHT,
                    self.BOARD_WIDTH,
                    image=screen_cv,
                    image_format="BGR",
                )
            else:
                self.board.set_hsv_image(screen_cv, image_format="BGR")

            self.board = self.board_analyzer.analyze_board(
                self.board, last_key_direction=self.current_direction
            )
            # 设置Board对象的snake_player引用
            self.board.snake_player = self

            game_state = self._determine_game_state()
            
            self.predict_or_update_head(self.board)
            self.analyze_time = time.time() - analyze_start
            return game_state, self.board

        except Exception as e:
            if self.logger:
                self.logger.error(f"[错误] 图像分析错误: {str(e)}")
            return None, None

    def _determine_game_state(self) -> str:
        """确定游戏状态"""
        if self.board_analyzer.is_gameover:
            return "game_over"
        elif self.board_analyzer.is_running:
            return "running"
        return "waiting"

    def _validate_board(self, board: Board) -> bool:
        """验证棋盘有效性"""
        if (
            not board
            or not hasattr(board, "cells")
            or not isinstance(board.cells, list)
            or not board.cells
        ):
            self.logger.error("analyze_board 返回棋盘非法")
            return False

        if (
            len(board.cells) < self.BOARD_HEIGHT
            or len(board.cells[0]) < self.BOARD_WIDTH
        ):
            self.logger.error(f"棋盘尺寸异常: {len(board.cells)}×{len(board.cells[0])}")
            return False

        return True

    def _evaluate_current_path(self):
        """评估当前路径"""
        try:
            self.current_path = self.update_and_evaluate_path(self.board)
        except Exception as e:
            self.logger.error(f"路径评估异常: {str(e)}")

    def _handle_running_state(self, hwnd):
        """处理游戏运行状态"""
        try:
            if not self.current_path or len(self.current_path) < 2:
                self.current_path = self.find_new_path()
            self.control_snake_by_path(hwnd)
        except Exception as e:
            self.logger.error(f"游戏运行状态处理异常: {str(e)}")

    def _handle_game_over_state(self, hwnd):
        """处理游戏结束状态"""
        if time.time() - self.last_click_time >= self.REVIVE_INTERVAL:
            try:
                self.snake_controller.set_game_window(hwnd)
                self.snake_controller.click_window_center()
                self.last_click_time = time.time()
            except Exception as e:
                self.logger.error("自动点击复活异常", e)

    def _emit_board_update(self, game_state: str):
        """发送棋盘更新信号"""
        try:
            self.board_updated.emit(
                game_state,
                self.board,
                self.convert_time,
                self.analyze_time,
                self.path_calc_time,
                self.draw_time,
                self.time_since_last_frame,
                self.current_direction,
                self.current_path,
            )
        except Exception as e:
            self.logger.error(f"信号emit异常: {str(e)}")

    def update_and_evaluate_path(self, board: Board) -> list:
        """评估并更新路径"""
        if not self.current_path:
            return self.current_path

        enemy_heads = [
            (c.col, c.row) for c in board.special_cells.get("enemy_head", [])
        ]

        # 检查当前路径是否存在危险
        for pos in self.current_path[0:6]:
            if not self._is_safe_position(board, pos, enemy_heads, force_mode=False):
                if self.logger:
                    self.logger.debug(f"[路径评估] 危险点: ({int(pos[0])+1},{int(pos[1])+1})")
                
                # 直接尝试寻找新路径
                new_path = self.find_new_path()
                if new_path:
                    if self.logger:
                        self.logger.debug(f"[路径评估] 新生成路径: {[(int(x)+1,int(y)+1) for x,y in new_path[0:6]]}")
                    return new_path
                
                if self.logger:
                    self.logger.debug("[路径评估] 无法找到安全路径，继续尝试")
                return self.find_new_path()

        return self.current_path

    def _is_safe_position(self, board: Board, pos: tuple, enemy_heads: list, force_mode: bool = False) -> bool:
        """检查位置是否安全
        
        Args:
            board: 棋盘对象
            pos: 位置坐标
            enemy_heads: 敌方蛇头列表
            force_mode: 强制模式（当为True时，只检查致命危险）
        """
        x, y = pos
        cell_type = board.cells[y][x].cell_type

        # 检查格子类型
        if cell_type not in self.SAFE_CELL_TYPES and cell_type != "predicted_body":
            self.logger.debug(
                f"[路径评估] 清空路径：遇到障碍物，位置({x},{y})，类型={cell_type}"
            )
            return False

        # 检查敌人距离
        for ex, ey in enemy_heads:
            distance = abs(x - ex) + abs(y - ey)
            if force_mode:
                # 强制模式下只检查是否会直接撞上
                if distance == 0:
                    self.logger.debug(
                        f"[路径评估] 强制模式：发现致命位置({x},{y})，与敌方蛇头({ex},{ey})重叠"
                    )
                    return False
            else:
                # 正常模式下保持安全距离
                if distance < 2:
                    self.logger.debug(
                        f"[路径评估] 清空路径：敌人蛇头({ex},{ey})距离路径点({x},{y})过近"
                    )
                    return False

        return True

    def control_snake_by_path(self, hwnd):
        """根据路径控制蛇移动"""
        if not self._prepare_snake_control(hwnd):
            if self.logger:
                self.logger.debug("[控制] 蛇控制准备失败，跳过移动")
            return

        # 检查是否需要执行缓存方向
        if self.cached_directions:
            if self.logger:
                self.logger.debug(f"[控制] 执行缓存方向: {self.cached_directions[0]}")
            self._execute_cached_directions(hwnd)
            return

        # 使用预测坐标代替实际观察坐标
        if not self.predicted_head_pos:
            real_head = self._get_snake_head()
            if not real_head:
                if self.logger:
                    self.logger.debug("[控制] 未找到蛇头位置")
                return
            head_pos = real_head
        else:
            head_pos = self.predicted_head_pos

        target = self._find_next_target(head_pos)
        if not target:
            if self.logger:
                self.logger.debug("[控制] 未找到下一个目标点")
            return

        direction = self._calculate_direction(head_pos, target)
        if direction:
            self.snake_move(
                hwnd,
                direction,
                reason=f"路径跟随：从({head_pos[0]},{head_pos[1]})到({target[0]},{target[1]})",
            )

    def _prepare_snake_control(self, hwnd) -> bool:
        """准备蛇的控制"""
        if not self.current_path:
            head_cell = self.board.special_cells.get("own_head", [None])[0]
            if head_cell:
                self.try_emergency_avoidance((head_cell.col, head_cell.row), hwnd)
            return False

        if len(self.current_path) < 2:
            return False

        return True

    def _get_snake_head(self) -> "tuple|None":
        """获取蛇头位置"""
        head_cell = self.board.special_cells.get("own_head", [None])[0]
        return (head_cell.col, head_cell.row) if head_cell else None

    def _find_next_target(self, real_head: tuple) -> "tuple|None":
        """找到下一个目标点
        如果蛇头与当前路径断开，尝试寻找一条连接到原路径起点的新路径
        """
        # 先尝试找相邻点
        for i, pos in enumerate(self.current_path):
            if self._is_adjacent(real_head, pos):
                self.current_path = self.current_path[i:]
                return pos
                
        # 如果没找到相邻点，检查附近是否有敌方蛇头
        enemy_heads = [(c.col, c.row) for c in self.board.special_cells.get("enemy_head", [])]
        for ex, ey in enemy_heads:
            if abs(real_head[0] - ex) + abs(real_head[1] - ey) < 3:  # 距离小于3格
                self.current_path = []
                if self.logger:
                    self.logger.debug(f"[路径] 发现附近敌方蛇头({ex},{ey})，清空路径")
                return None

        # 如果没找到相邻点，尝试连接到原路径起点
        if self.current_path:
            
            # 保存原路径
            original_path = self.current_path
            
            # 寻找连接路径
            connecting_path = self.path_finder.find_path(
                real_head,
                self.current_path[0],
                self.board,
                min_path_length=2,
                method="A"
            )
            
            if connecting_path:
                # 在拼接路径前，先释放所有按键
                self.snake_controller.control_snake(None)

                # 去掉连接路径的终点（因为它就是原路径的起点）
                connecting_path = connecting_path[:-1]

                # 拼接两条路径
                combined_path = connecting_path + original_path
                self.current_path = self._remove_backtracking(combined_path)

                # 行动前删除回头路
                return self.current_path[0]
        # 如果连接失败或没有原路径，清空路径
        self.current_path = []
        return None

    def _is_adjacent(self, pos1: tuple, pos2: tuple) -> bool:
        """检查两个点是否相邻"""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        return (dx == 1 and dy == 0) or (dx == 0 and dy == 1)

    def _calculate_direction(self, start: tuple, target: tuple) -> "str|None":
        """计算移动方向"""
        dx = target[0] - start[0]
        dy = target[1] - start[1]

        if dx == 1:
            return "right"
        if dx == -1:
            return "left"
        if dy == 1:
            return "down"
        if dy == -1:
            return "up"
        return None

    def snake_move(self, hwnd, direction: str, reason: str = ""):
        """控制蛇移动"""
        if not self._validate_move(direction):
            return

        now = time.time()
        if now - getattr(self, "last_snake_move_time", 0) < self.MOVE_INTERVAL:
            return

        if self._is_reverse_direction(direction):
            return

        self._execute_move(hwnd, direction, reason, now)

    def _validate_move(self, direction: str) -> bool:
        """验证移动有效性"""
        if (
            not isinstance(direction, str)
            or direction.lower() not in self.VALID_DIRECTIONS
        ):
            if self.logger:
                self.logger.error(f"无效direction参数 {direction}")
            return False
        return True

    def _is_reverse_direction(self, direction: str) -> bool:
        """检查是否反向移动"""
        if not self.current_direction:
            return False

        opposites = {"left": "right", "right": "left", "up": "down", "down": "up"}
        return opposites.get(self.current_direction) == direction

    def _execute_move(self, hwnd, direction: str, reason: str, now: float):
        """执行移动"""

        
        self.last_snake_move_time = now
        self.snake_controller.set_game_window(hwnd)
        self.snake_controller.control_snake(direction)
        self.current_direction = direction
        self.last_direction_time = now
        self._update_last_head_position()
        
        
    def _cache_future_directions(self):
        """缓存未来5步的方向"""
        if not self.current_path or len(self.current_path) < 2:
            self.cached_directions = []
            return
        
        self.last_execute_time = time.time()
        self.cached_directions = []
        for i in range(min(5, len(self.current_path)-1)):
            start = self.current_path[i]
            end = self.current_path[i+1]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            
            if dx == 1:
                self.cached_directions.append("right")
            elif dx == -1:
                self.cached_directions.append("left")
            elif dy == 1:
                self.cached_directions.append("down")
            elif dy == -1:
                self.cached_directions.append("up")
                
    def _execute_cached_directions(self, hwnd):
        """执行缓存的移动方向"""
        now = time.time()
        if now - self.last_execute_time < 0.19:
            return
            
        if not self.cached_directions:
            return
            
        # 检查路径有效性
        if self.board and self.board.special_cells.get("own_head"):
            head_cell = self.board.special_cells["own_head"][0]
            head_pos = (head_cell.col, head_cell.row)
            if not self._is_adjacent(head_pos, self.current_path[0]):
                self.cached_directions = []
                self.current_path = []
                if self.logger:
                    self.logger.debug("[路径] 路径与蛇头位置不连续，清空缓存路径")
                return
                
        direction = self.cached_directions.pop(0)

        # 执行移动后更新当前路径(移除已执行的路径点)
        if len(self.current_path) > 1:
            self.current_path = self.current_path[1:]

        self._execute_move(hwnd, direction, "执行缓存方向", now)
        self.last_execute_time = now

    def _update_last_head_position(self):
        """更新最后的蛇头位置"""
        if self.board and self.board.special_cells.get("own_head"):
            cell = self.board.special_cells["own_head"][0]
            self.last_time_control_snake_head = (cell.col, cell.row)
        else:
            self.last_time_control_snake_head = None

    def predict_or_update_head(self, board: Board):
        """预测或更新蛇头位置"""
        # 如果观察到蛇头，直接更新位置
        if board.special_cells.get("own_head"):
            head_cell = board.special_cells["own_head"][0]
            head_pos = (head_cell.col, head_cell.row)
            
            # 检查蛇头是否出现在预测蛇身中(排除第一个位置)
            if len(self.predicted_body_positions) > 1 and head_pos in self.predicted_body_positions[1:]:
                # 如果蛇头出现在预测蛇身中，说明可能是检测抖动，强制使用预测位置
                if self.predicted_body_positions:
                    head_pos = self.predicted_body_positions[0]
                    if self.logger:
                        self.logger.debug(f"[预测] 检测到蛇头抖动，强制使用预测位置: {head_pos}")
            
            self.observed_head_pos = head_pos
            self.predicted_head_pos = head_pos
            self.last_time_control_snake_head = head_pos
            
            # 更新预测蛇身位置
            if not self.predicted_body_positions or self.predicted_body_positions[0] != head_pos:
                self.predicted_body_positions.insert(0, head_pos)
                # 限制预测蛇身长度，最多保持3格
                if len(self.predicted_body_positions) > 3:
                    self.predicted_body_positions.pop()
            
            # 更新棋盘上的预测蛇身
            self._update_board_with_predicted_body(board)
        else:
            # 未观察到蛇头时才进行预测
            if self._can_predict_head():
                predicted_pos = self._calculate_predicted_position()
                if predicted_pos:
                    self.predicted_head_pos = predicted_pos
                    # 使用预测位置更新棋盘
                    self._update_board_with_predicted_head(board, predicted_pos)
                # 更新预测蛇身到棋盘
                self._update_board_with_predicted_body(board)
                
        # 预测后检测是否需要缓存未来5步方向
        if self._should_cache_future_moves(board):
            self._cache_future_directions()

    def _should_cache_future_moves(self, board: Board) -> bool:
        """检测是否需要缓存未来5步方向"""
        if not self.current_path or len(self.current_path) < 6:
            return False

        if len(self.cached_directions) >= 1:
            return False
            
        # 检查转弯次数是否超过3次
        turn_count = 0
        last_direction = None
        for i in range(5):
            start = self.current_path[i]
            end = self.current_path[i+1]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            
            current_direction = None
            if dx == 1:
                current_direction = "right"
            elif dx == -1:
                current_direction = "left"
            elif dy == 1:
                current_direction = "down"
            elif dy == -1:
                current_direction = "up"
                
            if last_direction and current_direction != last_direction:
                turn_count += 1
                if turn_count > 3:
                    return True
            last_direction = current_direction
            
        # 检查下一格是否是分数格
        next_pos = self.current_path[1]
        cell = board.cells[next_pos[1]][next_pos[0]]
        if cell.cell_type == "score_boost":
            return True
            
        return False
        
    def _can_predict_head(self) -> bool:
        """检查是否可以预测蛇头"""
        return bool(
            self.last_time_control_snake_head
            and self.current_direction
            and self.last_direction_time
        )

    def _calculate_predicted_position(self) -> "tuple|None":
        """计算预测位置"""
        elapsed_time = time.time() - self.last_direction_time
        steps = int(elapsed_time / 0.2)
        x, y = self.last_time_control_snake_head

        direction_vectors = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
        }
        dx, dy = direction_vectors.get(self.current_direction, (0, 0))

        px = x + steps * dx
        py = y + steps * dy

        if not (0 <= px < self.BOARD_WIDTH and 0 <= py < self.BOARD_HEIGHT):
            return None

        return (px, py)

    def _update_board_with_predicted_head(self, board: Board, pos: tuple):
        """使用预测位置更新棋盘"""
        cell = board.cells[pos[1]][pos[0]]
        cell.cell_type = "own_head"

        if "own_head" not in board.special_cells:
            board.special_cells["own_head"] = []

        board.special_cells["own_head"] = [
            c
            for c in board.special_cells["own_head"]
            if not (c.row == cell.row and c.col == cell.col)
        ]
        board.special_cells["own_head"].append(cell)
        
    def _update_board_with_predicted_body(self, board: Board):
        """使用预测蛇身位置更新棋盘
        从距离蛇头最近的蛇身开始更新，遇到蛇尾则停止
        """
        # 确保预测蛇身类型存在于special_cells中
        if "predicted_body" not in board.special_cells:
            board.special_cells["predicted_body"] = []
        
        # 清空之前的预测蛇身
        board.special_cells["predicted_body"] = []
        
        # 从距离蛇头最近的蛇身开始更新
        for pos in self.predicted_body_positions:
            x, y = pos
            # 确保坐标在棋盘范围内
            if 0 <= x < self.BOARD_WIDTH and 0 <= y < self.BOARD_HEIGHT:
                cell = board.cells[y][x]
                
                # 遇到蛇尾则停止更新
                if cell.cell_type == "own_tail":
                    break
                else:
                    cell.cell_type = "own_body"
                    board.special_cells["predicted_body"].append(cell)

    def find_new_path(self, emergency: bool = False):
        """寻找新路径"""
        self._calc_start = time.time()

        if not emergency:
            path = self._try_normal_path()
            if path:
                return path

        path = self._try_escape_path()
        if path:
            return path

        return self.current_path

    def _try_normal_path(self) -> "list|None":
        """尝试常规路径"""
        best_path = self.path_finder.find_path_in_order(self.board)
        if best_path and len(best_path) > 1:
            return self._update_path_and_emit(best_path)
        return None

    def _try_escape_path(self) -> "list|None":
        """尝试逃生路径"""
        escape_path = self.path_finder.find_path_to_nearest(self.board)
        if escape_path and len(escape_path) > 1:
            return self._update_path_and_emit(escape_path)
        return None

    def _update_path_and_emit(self, path: list) -> list:
        """更新路径并发送信号"""
        self.current_path = path
        self._emit_path_update()
        self.path_calc_time = time.time() - self._calc_start
        return self.current_path

    def _emit_path_update(self):
        """发送路径更新信号"""
        self.board_updated.emit(
            "running",
            self.board,
            self.convert_time,
            self.analyze_time,
            self.path_calc_time,
            self.draw_time,
            self.time_since_last_frame,
            self.current_direction,
            self.current_path,
        )

    def try_emergency_avoidance(self, head_pos: tuple, hwnd):
        """紧急避险
        Args:
            head_pos: 蛇头位置(x, y)
            hwnd: 窗口句柄
        """
        # 获取周围可用方向
        available_directions = []
        x, y = head_pos
        directions = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

        # 检查四个方向
        for direction, (dx, dy) in directions.items():
            new_x, new_y = x + dx, y + dy
            # 检查是否在边界内
            if 0 <= new_x < self.BOARD_WIDTH and 0 <= new_y < self.BOARD_HEIGHT:
                # 检查该位置是否安全
                if self.board.cells[new_y][new_x].cell_type in self.SAFE_CELL_TYPES:
                    available_directions.append(direction)

        # 如果有可用方向，随机选择一个
        if available_directions:
            direction = available_directions[0]  # 选择第一个可用方向
            self.snake_move(hwnd, direction, reason="紧急避险")

    def predict_snake_position(self) -> "tuple|None":
        """预测小蛇当前坐标
        基于最后控制方向和最后控制时间计算预测位置
        
        Returns:
            tuple|None: 返回预测的(x,y)坐标，如果无法预测则返回None
        """
        if not self._can_predict_head():
            return None
            
        predicted_pos = self._calculate_predicted_position()
        if predicted_pos:
            x, y = predicted_pos
            if 0 <= x < self.BOARD_WIDTH and 0 <= y < self.BOARD_HEIGHT:
                return predicted_pos
        return None
        
    def _remove_backtracking(self, path: list) -> list:
        """删除路径中的回头路
        参数:
            path: 原始路径列表 [(x1,y1), (x2,y2), ...]
        返回:
            清理后的路径列表
        """
        if len(path) < 3:
            return path
            
        cleaned_path = [path[0]]
        i = 1
        
        while i < len(path):
            # 检查当前点与前一个点和后一个点是否形成回头路
            if (i + 1 < len(path) and 
                path[i-1] == path[i+1]):
                # 跳过回头路的中间点
                i += 2
            else:
                cleaned_path.append(path[i])
                i += 1
                
        return cleaned_path
