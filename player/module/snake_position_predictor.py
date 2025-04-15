import numpy as np
from player.module.logging_mixin import LoggingMixin
from model.snake_board import Board

class SnakePositionPredictor(LoggingMixin):
    """
    蛇位置预测器：负责蛇头和蛇身位置的预测
    主要功能：
    1. 蛇头位置预测
    2. 蛇身位置预测
    3. 碰撞检测
    """
    
    # 常量定义
    BOARD_WIDTH = 29
    BOARD_HEIGHT = 25
    
    def __init__(self, logger=None):
        super().__init__(logger)
        # 蛇头位置相关
        self.observed_head_pos = None  # 实际观察到的蛇头位置
        self.predicted_head_pos = None  # 预测的蛇头位置
        self.predicted_body_positions = []  # 预测的蛇身位置列表(最多保持5格)
        self.last_time_control_snake_head = None  # 上次控制蛇头的时间
    
    def predict_or_update_head(self, board: Board):
        """
        预测或更新蛇头位置
        如果观察到蛇头，则更新；否则使用预测位置
        """
        if board.head_position:
            # 观察到蛇头，更新位置
            self.observed_head_pos = board.head_position
            self.predicted_head_pos = board.head_position
            self.log_debug(f"[位置] 观察到蛇头位置: {self.observed_head_pos}")
            
            # 更新蛇身预测位置
            self._update_predicted_body_positions(board)
        elif self.predicted_head_pos:
            # 未观察到蛇头，使用预测位置
            self.log_debug(f"[位置] 使用预测蛇头位置: {self.predicted_head_pos}")
            # 设置预测的蛇头位置到棋盘
            x, y = self.predicted_head_pos
            if 0 <= y < len(board.cells) and 0 <= x < len(board.cells[0]):
                board.cells[y][x].cell_type = "predicted_head"
                # 添加到特殊单元格列表
                if "predicted_head" not in board.special_cells:
                    board.special_cells["predicted_head"] = []
                board.special_cells["predicted_head"].append(board.cells[y][x])
            
            # 设置预测的蛇身位置到棋盘
            for pos in self.predicted_body_positions:
                x, y = pos
                if 0 <= y < len(board.cells) and 0 <= x < len(board.cells[0]):
                    board.cells[y][x].cell_type = "predicted_body"
                    # 添加到特殊单元格列表
                    if "predicted_body" not in board.special_cells:
                        board.special_cells["predicted_body"] = []
                    board.special_cells["predicted_body"].append(board.cells[y][x])
    
    def _update_predicted_body_positions(self, board: Board):
        """
        更新预测的蛇身位置
        根据观察到的蛇身和蛇头位置，更新预测的蛇身位置列表
        """
        # 清空预测的蛇身位置列表
        self.predicted_body_positions = []
        
        # 获取观察到的蛇身位置
        observed_body_positions = [(cell.col, cell.row) for cell in board.special_cells.get("own_body", [])]
        
        # 如果有观察到的蛇身，使用观察到的蛇身位置
        if observed_body_positions:
            # 只保留最近的5个蛇身位置
            self.predicted_body_positions = observed_body_positions[:5]
            self.log_debug(f"[位置] 更新预测蛇身位置: {self.predicted_body_positions}")
    
    def predict_next_head_position(self, direction: str):
        """
        根据当前方向预测下一个蛇头位置
        """
        if not self.predicted_head_pos or not direction:
            return None
        
        x, y = self.predicted_head_pos
        
        # 根据方向计算下一个位置
        if direction == "up":
            y = max(0, y - 1)
        elif direction == "down":
            y = min(self.BOARD_HEIGHT - 1, y + 1)
        elif direction == "left":
            x = max(0, x - 1)
        elif direction == "right":
            x = min(self.BOARD_WIDTH - 1, x + 1)
        else:
            self.log_error(f"无效的方向: {direction}")
            return None
        
        return (x, y)
    
    def update_predicted_positions(self, direction: str, current_time: float):
        """
        更新预测的蛇头和蛇身位置
        """
        if not self.predicted_head_pos:
            return
        
        # 记录当前蛇头位置作为蛇身
        self.predicted_body_positions.insert(0, self.predicted_head_pos)
        # 只保留最近的5个蛇身位置
        self.predicted_body_positions = self.predicted_body_positions[:5]
        
        # 预测下一个蛇头位置
        next_head_pos = self.predict_next_head_position(direction)
        if next_head_pos:
            self.predicted_head_pos = next_head_pos
            self.last_time_control_snake_head = current_time
            self.log_debug(f"[位置] 更新预测蛇头位置: {self.predicted_head_pos}")
    
    def check_collision(self, board: Board, position: tuple) -> bool:
        """
        检查位置是否会发生碰撞
        """
        if not position:
            return True
        
        x, y = position
        # 检查是否超出边界
        if x < 0 or x >= self.BOARD_WIDTH or y < 0 or y >= self.BOARD_HEIGHT:
            return True
        
        # 检查是否碰到障碍物
        cell_type = board.cells[y][x].cell_type
        safe_cell_types = {
            "empty", "score_boost", "own_head", "unknown", 
            "own_tail", "predicted_head", "predicted_body"
        }
        
        return cell_type not in safe_cell_types