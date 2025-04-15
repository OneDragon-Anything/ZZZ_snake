from collections import deque
import random
import time
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

# 导入重构后的模块
from analyzer.path.pathfinding_algorithms import AStarAlgorithm, BFSAlgorithm
from analyzer.path.risk_analyzer import RiskAnalyzer
from analyzer.path.space_analyzer import SpaceAnalyzer


class PathFinder(QObject):
    """路径查找器类，用于在网格中寻找路径
    重构后的版本将核心算法和功能拆分到专门的模块中，提高代码可维护性
    """

    # 定义信号
    path_updated = pyqtSignal(object, object)  # 当前路径、目标位置

    def __init__(self, grid_height, grid_width, logger=None):
        """初始化路径查找器
        
        Args:
            grid_height: 网格高度
            grid_width: 网格宽度
            logger: 日志记录器对象，默认为None
        """
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.start_pos = None
        self.current_direction = "right"
        self.logger = logger
        
        # 初始化各个分析器和算法
        self.risk_analyzer = RiskAnalyzer(grid_height, grid_width)
        self.space_analyzer = SpaceAnalyzer(grid_height, grid_width)
        self.astar = AStarAlgorithm(grid_height, grid_width, self.risk_analyzer.risk_array)
        self.bfs = BFSAlgorithm(grid_height, grid_width, self.risk_analyzer.risk_array)
        
        # 为了兼容性保留风险分数字典的引用
        self.risk_scores = self.risk_analyzer.risk_scores
        self.risk_array = self.risk_analyzer.risk_array

    def update_risk_areas(self, board):
        """更新风险区域
        
        Args:
            board: 棋盘对象
        """
        self.risk_analyzer.update_risk_areas(board)
        # 更新引用，确保兼容性
        self.risk_scores = self.risk_analyzer.risk_scores

    def _add_risk_score(self, pos, score):
        """辅助方法：累加风险分数
        
        Args:
            pos: 位置坐标 (x, y)
            score: 风险分数
        """
        self.risk_analyzer.add_risk_score(pos, score)

    def find_path(self, start, target, board, init_path=None, min_path_length=2, method="A"):
        """统一寻路接口：支持A*和BFS
        
        Args:
            start: 起点坐标 (x, y)
            target: 目标坐标 (x, y)
            board: 棋盘对象
            init_path: 初始路径，用于断点续寻
            min_path_length: 最小路径长度
            method: 寻路方法，'A'用A*，'B'用BFS，默认'A'
            
        Returns:
            路径列表或None
        """
        t0 = time.time()
        
        # 确保风险区域是最新的
        self.update_risk_areas(board)
        
        if method == "A":
            path = self.astar.find_path(
                start,
                target,
                board,
                init_path=init_path,
                min_path_length=min_path_length,
            )
        else:
            path = self.bfs.find_path(
                start,
                target,
                board,
                init_path=init_path,
                min_path_length=min_path_length,
            )
        t1 = time.time()
        
        if self.logger and path:
            self.logger.debug(f"[寻路] 方法:{method} 耗时:{(t1-t0)*1000:.1f}ms 路径长度:{len(path)}")

        return path

    def find_path_to_score_boost(self, board, direction=None):
        """尝试往分数点，再接安全区域，确保安全
        
        Args:
            board: 当前棋盘
            direction: 当前方向
            
        Returns:
            路径列表或None
        """
        self.current_direction = direction if direction else board.direction
        if "own_head" not in board.special_cells or not board.special_cells["own_head"]:
            return None
        head_cell = board.special_cells["own_head"][0]
        self.start_pos = (head_cell.col, head_cell.row)

        score_boosts = []
        if "score_boost" in board.special_cells:
            boost_cells = board.special_cells["score_boost"]
            if boost_cells:
                # 提取所有分数点坐标
                boost_coords = np.array([(cell.col, cell.row) for cell in boost_cells])
                
                # 为每个分数点创建安全标志
                safe_flags = np.ones(len(boost_coords), dtype=bool)
                
                # 创建周围1格的偏移量数组
                offsets = np.array([(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2)])
                
                # 对每个分数点检查安全性
                for i, (x, y) in enumerate(boost_coords):
                    # 计算周围所有格子的坐标
                    surrounding = np.array([x, y]) + offsets
                    
                    # 检查边界条件
                    valid = (surrounding[:, 0] >= 0) & (surrounding[:, 0] < board.cols) & \
                            (surrounding[:, 1] >= 0) & (surrounding[:, 1] < board.rows)
                    
                    # 如果有任何格子超出边界，标记为不安全
                    if not np.all(valid):
                        safe_flags[i] = False
                        continue
                    
                    # 检查有效格子的类型
                    for sx, sy in surrounding[valid]:
                        cell_type = board.cells[sy][sx].cell_type
                        if cell_type not in [
                            "empty",
                            "own_head",
                            "own_body",
                            "own_tail",
                            "score_boost",
                        ]:
                            safe_flags[i] = False
                            break
                
                # 收集所有安全的分数点
                safe_boosts = boost_coords[safe_flags]
                score_boosts = [tuple(coord) for coord in safe_boosts]
                
                # 记录不安全的分数点
                if self.logger:
                    unsafe_boosts = boost_coords[~safe_flags]
                    for x, y in unsafe_boosts:
                        self.logger.debug(
                            f"[寻路] 分数点({x},{y})周围3格内存在危险或超出边界，跳过"
                        )

        # 没有分数点时直接返回None
        if not score_boosts:
            return None

        for target in score_boosts:
            # 确保目标点是有效的
            if not (0 <= target[0] < board.cols and 0 <= target[1] < board.rows):
                continue

            path1 = self.find_path(
                self.start_pos, target, board, method="A", min_path_length=3
            )
            if path1:
                if path1[-1] != target:
                    continue

                # 找到去score的路后，尝试接上安全区域
                # 获取所有矩形中心点，按面积从大到小排序
                centers = self.space_analyzer.find_largest_empty_rectangle(board)
                
                # 尝试所有矩形中心点，从大到小
                for center_x, center_y in centers:
                    safe_target = (center_x, center_y)
                    path2 = self.find_path(
                        None,
                        safe_target,
                        board,
                        init_path=path1,
                        min_path_length=5,
                        method="A",
                    )
                    if path2:
                        return path2

                # 如果没有找到连接到安全区域的路径，至少返回到分数点的路径
                return path1

        return None

    def find_safe_path(self, board):
        """智能逃生策略：
        1. 尝试所有空白矩形中心点（按面积从大到小）
        2. 尝试距离敌方蛇头最远的点
        3. 尝试任意可达的空白点
        
        Args:
            board: 棋盘对象
            
        Returns:
            路径列表或None
        """
        # 1. 尝试所有矩形中心点
        centers = self.space_analyzer.find_largest_empty_rectangle(board)
        for center_x, center_y in centers:
            # 检查中心点是否可通行
            if 0 <= center_x < board.cols and 0 <= center_y < board.rows:
                cell = board.cells[center_y][center_x]
                if cell and cell.cell_type == "empty":
                    path = self.find_path(
                        self.start_pos,
                        (center_x, center_y),
                        board,
                        min_path_length=3,
                        method="A",  # 优先使用A*算法
                    )
                    if path:
                        return path

        # 3. 尝试寻找最近出路
        return self.find_path_to_nearest(board)

    def find_path_to_tail(self, board):
        """计算去往自己尾部的路径，用于防止围死自己
        
        Args:
            board: 棋盘对象
            
        Returns:
            路径列表或None
        """
        self.current_direction = board.direction
        if "own_head" not in board.special_cells or not board.special_cells["own_head"]:
            return None
        head_cell = board.special_cells["own_head"][0]
        head_pos = (head_cell.col, head_cell.row)

        self.start_pos = head_pos

        if "own_tail" not in board.special_cells:
            return None
        tail_cell = board.special_cells["own_tail"][0]
        tail_pos = (tail_cell.col, tail_cell.row)

        path = self.find_path(head_pos, tail_pos, board, min_path_length=5, method="A")
        if path:
            return path

        path = self.find_path(head_pos, tail_pos, board, min_path_length=3, method="A")
        if path:
            return path

        return None

    def find_path_to_nearest(self, board):
        """基于风险评估的安全路径寻找
        直接寻找一条风险最小的可行路径
        
        Args:
            board: 棋盘对象
            
        Returns:
            路径列表或None
        """
        if "own_head" not in board.special_cells or not board.special_cells["own_head"]:
            return None
        head_cell = board.special_cells["own_head"][0]
        self.start_pos = (head_cell.col, head_cell.row)
        
        rows, cols = board.rows, board.cols
        
        # 找出所有可能的目标点
        targets = []
        for y in range(rows):
            for x in range(cols):
                cell = board.cells[y][x]
                if cell and cell.cell_type == "empty":
                    # 计算安全度（风险值越高安全度越低）
                    risk = self.risk_analyzer.get_risk_score(x, y)
                    if risk < 5:  # 只考虑低风险区域
                        targets.append((x, y))
        
        # 按照到蛇头的距离排序目标点
        targets.sort(key=lambda p: abs(p[0] - self.start_pos[0]) + abs(p[1] - self.start_pos[1]))
        
        # 尝试寻找到每个候选点的路径
        for x, y in targets:
            path = self.find_path(
                self.start_pos,
                (x, y),
                board,
                min_path_length=3,
                method="A"
            )
            if path:
                return path
                
        return None

    def find_escape_route(self, board):
        """极速逃生策略：在面临直接碰撞风险时，快速选择一个最不坏的邻近格子。
        优先考虑生存，其次考虑低风险和开阔度。
        
        Args:
            board: 棋盘对象
            
        Returns:
            一个包含当前位置和下一步位置的列表 [current_pos, next_pos]，或 None
        """
        if not hasattr(board, 'special_cells') or "own_head" not in board.special_cells:
            return None
            
        head_cell = board.special_cells["own_head"][0]
        self.start_pos = (head_cell.col, head_cell.row)
        start_x, start_y = self.start_pos

        # 获取当前方向向量
        current_direction_vector = self.space_analyzer.get_current_direction(board, self.start_pos)

        # 定义移动方向 (dx, dy)
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 上下左右
        possible_next_steps = []  # 存储 (得分, (nx, ny))

        # 遍历所有邻居
        for dx, dy in moves:
            # 排除向后移动
            if current_direction_vector and (dx, dy) == (-current_direction_vector[0], -current_direction_vector[1]):
                continue

            next_x, next_y = start_x + dx, start_y + dy
            next_pos = (next_x, next_y)

            # 检查边界
            if not (0 <= next_x < board.cols and 0 <= next_y < board.rows):
                continue

            # 检查障碍物
            cell = board.cells[next_y][next_x]
            cell_type = cell.cell_type if cell else "wall"

            # 绝对不能走的格子
            lethal_types = {"enemy_head", "enemy_body", "mine", "own_body"}
            if cell_type in lethal_types:
                continue

            # 评分系统
            score = 50  # 基础生存分

            # 格子类型加分
            type_scores = {
                "empty": 20,
                "score_boost": 25,
                "own_tail": 10
            }
            score += type_scores.get(cell_type, 0)

            # 风险评估
            risk = self.risk_analyzer.get_risk_score(next_x, next_y)
            score -= risk * 2

            # 评估开阔度
            openness = sum(1 for ndx, ndy in moves
                         if 0 <= next_x + ndx < board.cols
                         and 0 <= next_y + ndy < board.rows
                         and board.cells[next_y + ndy][next_x + ndx].cell_type
                         in ["empty", "score_boost", "own_tail"])
            score += openness * 5

            possible_next_steps.append((score, next_pos))

        if not possible_next_steps:
            if self.logger:
                self.logger.warning(f"[紧急逃生] 位置 ({start_x},{start_y}) 周围无路可走!")
            return None

        # 选择最佳方向
        possible_next_steps.sort(key=lambda item: item[0], reverse=True)
        best_next_pos = possible_next_steps[0][1]
        
        if self.logger:
            self.logger.info(f"[紧急逃生] 从 ({start_x},{start_y}) 选择逃向 {best_next_pos}，得分: {possible_next_steps[0][0]}")

        return [self.start_pos, best_next_pos]

    def find_path_in_order(self, board):
        """优先策略顺序寻路（分数道具优先，再找尾巴，再瞎走）
        当蛇头周围1格内存在中高风险或地图边缘时，优先使用极速逃生策略
        
        Args:
            board: 棋盘对象
            
        Returns:
            路径列表或None
        """
        # 确保有蛇头信息
        if "own_head" not in board.special_cells or not board.special_cells["own_head"]:
            return None
        head_cell = board.special_cells["own_head"][0]
        self.start_pos = (head_cell.col, head_cell.row)
        
        # 检查蛇头周围1格是否存在危险或地图边缘
        x, y = self.start_pos
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 上下左右
        for dx, dy in moves:
            next_x, next_y = x + dx, y + dy
            # 检查是否在地图边缘
            if not (0 <= next_x < board.cols and 0 <= next_y < board.rows):
                # 发现地图边缘，使用极速逃生
                if self.logger:
                    self.logger.info(f"[寻路] 蛇头({x},{y})周围1格发现地图边缘，启动极速逃生")
                return self.find_escape_route(board)
            
            # 检查是否有敌方蛇头/蛇身或地雷
            cell = board.cells[next_y][next_x]
            cell_type = cell.cell_type if cell else None
            if cell_type in ["enemy_head", "enemy_body", "mine"]:
                # 发现危险，使用极速逃生
                if self.logger:
                    self.logger.info(f"[寻路] 蛇头({x},{y})周围1格发现危险({cell_type})，启动极速逃生")
                return self.find_escape_route(board)

        path = self.find_path_to_score_boost(board)
        if path:
            return path

        path = self.find_safe_path(board)
        if path:
            return path

        path = self.find_path_to_tail(board)
        if path:
            return path

        return None