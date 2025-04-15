import time
from player.module.logging_mixin import LoggingMixin
from model.snake_board import Board
from analyzer.path.path_finder import PathFinder

class PathEvaluator(LoggingMixin):
    """
    路径评估器：负责路径评估和优化
    主要功能：
    1. 路径安全性检查
    2. 路径优化
    3. 紧急避险策略
    """
    
    # 常量定义
    BOARD_WIDTH = 29
    BOARD_HEIGHT = 25
    PATH_CALC_INTERVAL = 0.2
    SAFE_CELL_TYPES = {
        "empty", "score_boost", "own_head", "unknown", 
        "own_tail", "own_body", "predicted_head", "predicted_body"
    }
    
    def __init__(self, logger=None):
        super().__init__(logger)
        self.path_finder = PathFinder(self.BOARD_HEIGHT, self.BOARD_WIDTH, logger)
        self.current_path = []
        self.last_path_calc_time = 0
        self.path_calc_time = 0
        self.reason = None  # 记录路径评估原因
    
    def find_new_path(self, board: Board) -> list:
        """
        寻找新路径
        """
        if not board or not board.head_position:
            self.log_error("寻找新路径失败：无效的棋盘或蛇头位置")
            return []
        
        # 检查是否需要重新计算路径（避免频繁计算）
        current_time = time.time()
        if current_time - self.last_path_calc_time < self.PATH_CALC_INTERVAL and self.current_path:
            return self.current_path
        
        path_calc_start = time.time()
        try:
            # 获取目标点
            target_cells = board.special_cells.get("score_boost", [])
            if not target_cells:
                self.log_debug("未找到得分点，使用默认目标")
                # 如果没有得分点，使用默认目标（棋盘中心）
                self.current_path = self._find_path_to_center(board)
            else:
                self.current_path = self._find_path_to_nearest_target(board, target_cells)
            
            self.last_path_calc_time = current_time
            self.path_calc_time = time.time() - path_calc_start
            return self.current_path
        
        except Exception as e:
            self.log_error("寻找新路径异常", e)
            self.path_calc_time = time.time() - path_calc_start
            return []
    
    def _find_path_to_center(self, board: Board) -> list:
        """
        寻找到棋盘中心的路径
        """
        center_x = self.BOARD_WIDTH // 2
        center_y = self.BOARD_HEIGHT // 2
        # 修正参数顺序：start, target, board
        return self.path_finder.find_path(board.head_position, (center_x, center_y), board)
    
    def _find_path_to_nearest_target(self, board: Board, target_cells: list) -> list:
        """
        寻找到最近目标的路径
        """
        if not target_cells:
            return []
        

        # 获取所有目标点的坐标
        targets = [(cell.col, cell.row) for cell in target_cells]
        
        # 找到最近的目标点
        best_path = []
        min_distance = float('inf')
        
        for target in targets:
            # 修正参数顺序：start, target, board
            path = self.path_finder.find_path(board.head_position, target, board)
            if path and len(path) < min_distance:
                min_distance = len(path)
                best_path = path
        
        return best_path
    
    def evaluate_path(self, board: Board, current_path: list) -> list:
        """
        评估当前路径的安全性，如果不安全则寻找新路径
        """
        if not current_path:
            return self.find_new_path(board)
        
        # 获取敌方蛇头位置
        enemy_heads = [(c.col, c.row) for c in board.special_cells.get("enemy_head", [])]
        
        # 检查当前路径前几步是否安全
        for pos in current_path[0:6]:
            if not self._is_safe_position(board, pos, enemy_heads, force_mode=False):
                # 尝试寻找新路径
                new_path = self.find_new_path(board)
                if new_path:
                    # 检查新路径是否也存在危险
                    has_danger = False
                    for pos in new_path[0:6]:
                        if not self._is_safe_position(board, pos, enemy_heads, force_mode=False):
                            has_danger = True
                            break
                    
                    if has_danger:
                        # 新路径也有危险，说明可能是唯一路径
                        # 重新检查新路径（使用强制模式）
                        is_viable = True
                        for pos in new_path[0:6]:
                            if not self._is_safe_position(board, pos, enemy_heads, force_mode=True):
                                is_viable = False
                                break
                        if is_viable:
                            self.log_debug("[路径评估] 发现唯一可行路径，虽有风险但必须尝试")
                            return new_path
                    else:
                        # 新路径安全，使用新路径
                        return new_path
                return self.find_new_path(board)
        
        return current_path
    
    def _is_safe_position(self, board: Board, pos: tuple, enemy_heads: list, force_mode: bool = False) -> bool:
        """
        检查位置是否安全
        
        Args:
            board: 棋盘对象
            pos: 位置坐标
            enemy_heads: 敌方蛇头列表
            force_mode: 强制模式（当为True时，只检查致命危险）
        """
        x, y = pos
        
        # 检查是否超出边界
        if x < 0 or x >= self.BOARD_WIDTH or y < 0 or y >= self.BOARD_HEIGHT:
            self.reason = f"位置({x},{y})超出边界"
            return False
        
        # 获取格子类型
        cell_type = board.cells[y][x].cell_type
        
        # 检查格子类型
        if cell_type not in self.SAFE_CELL_TYPES:
            self.reason = f"位置({x},{y})存在障碍物，类型={cell_type}"
            return False
        
        # 在强制模式下，只检查致命危险
        if force_mode:
            return True
        
        # 检查是否靠近敌方蛇头（可能被吃）
        for ex, ey in enemy_heads:
            if abs(x - ex) <= 2 and abs(y - ey) <= 2:
                self.reason = f"位置({x},{y})靠近敌方蛇头({ex},{ey})"
                return False
        
        return True
    
    def get_direction_from_path(self, head_pos: tuple, path: list) -> str:
        """
        根据路径获取下一步方向
        """
        if not path or len(path) < 2 or not head_pos:
            return ""
        
        # 获取下一个位置
        next_pos = path[1] if path[0] == head_pos else path[0]
        
        # 计算方向
        x1, y1 = head_pos
        x2, y2 = next_pos
        
        if x1 == x2 and y1 > y2:
            return "up"
        elif x1 == x2 and y1 < y2:
            return "down"
        elif x1 > x2 and y1 == y2:
            return "left"
        elif x1 < x2 and y1 == y2:
            return "right"
        
        # 如果无法确定方向，返回空字符串
        return ""