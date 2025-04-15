import time
from PyQt5.QtCore import QObject, pyqtSignal
from player.module.logging_mixin import LoggingMixin
from model.snake_board import Board

class GameStateManager(LoggingMixin, QObject):
    """
    游戏状态管理器：负责游戏状态检测和管理
    主要功能：
    1. 游戏状态检测
    2. 状态转换
    3. 信号发送
    """
    
    # 状态更新信号：游戏状态、棋盘、各阶段耗时、当前方向、路径
    board_updated = pyqtSignal(
        object, object, float, float, float, float, float, str, object
    )
    
    # 常量定义
    REVIVE_INTERVAL = 1.0
    
    def __init__(self, logger=None):
        QObject.__init__(self)
        LoggingMixin.__init__(self, logger)
        self.last_click_time = 0
        self.last_process_time = None
        self.time_since_last_frame = 0
        
        # 性能统计相关
        self.convert_time = 0
        self.analyze_time = 0
        self.path_calc_time = 0
        self.draw_time = 0
    
    def update_frame_time(self):
        """
        更新帧时间
        """
        now = time.time()
        self.time_since_last_frame = (
            0 if self.last_process_time is None else now - self.last_process_time
        )
        self.last_process_time = now
    
    def determine_game_state(self, board_analyzer) -> str:
        """
        确定游戏状态
        """
        if board_analyzer.is_gameover:
            return "game_over"
        elif board_analyzer.is_running:
            return "running"
        return "waiting"
    
    def handle_game_over(self, hwnd, snake_controller):
        """
        处理游戏结束状态
        """
        if time.time() - self.last_click_time >= self.REVIVE_INTERVAL:
            try:
                snake_controller.set_game_window(hwnd)
                snake_controller.click_window_center()
                self.last_click_time = time.time()
                self.log_debug("[状态] 游戏结束，点击重新开始")
            except Exception as e:
                self.log_error("自动点击复活异常", e)
    
    def emit_board_update(self, game_state: str, board: Board, 
                          current_direction: str, current_path: list,
                          convert_time: float, analyze_time: float, 
                          path_calc_time: float, draw_time: float):
        """
        发送棋盘更新信号
        """
        try:
            self.board_updated.emit(
                game_state,
                board,
                convert_time,
                analyze_time,
                path_calc_time,
                draw_time,
                self.time_since_last_frame,
                current_direction,
                current_path,
            )
        except Exception as e:
            self.log_error("信号emit异常", e)
    
    def validate_board(self, board: Board, board_height: int, board_width: int) -> bool:
        """
        验证棋盘有效性
        """
        if (
            not board
            or not hasattr(board, "cells")
            or not isinstance(board.cells, list)
            or not board.cells
        ):
            self.log_error("analyze_board 返回棋盘非法")
            return False

        if (
            len(board.cells) < board_height
            or len(board.cells[0]) < board_width
        ):
            self.log_error(f"棋盘尺寸异常: {len(board.cells)}×{len(board.cells[0])}")
            return False

        return True