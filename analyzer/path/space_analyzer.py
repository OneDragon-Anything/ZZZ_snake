import numpy as np
from collections import deque

class SpaceAnalyzer:
    """空间分析器，负责分析网格中的空间结构和可达性"""
    
    def __init__(self, grid_height, grid_width):
        """初始化空间分析器
        
        Args:
            grid_height: 网格高度
            grid_width: 网格宽度
        """
        self.grid_height = grid_height
        self.grid_width = grid_width
    
    def find_largest_empty_rectangle(self, board):
        """使用基于"柱状图中最大矩形"的优化算法寻找所有空白矩形。
        
        Args:
            board: 棋盘对象
            
        Returns:
            按面积从大到小排序的矩形中心坐标列表 [(x1,y1), (x2,y2), ...]
        """
        rows, cols = board.rows, board.cols
        if rows == 0 or cols == 0:
            return []

        # 预计算空白掩码，提高后续访问速度
        empty_mask = np.zeros((rows, cols), dtype=bool)
        for r in range(rows):
            for c in range(cols):
                cell = board.cells[r][c]
                # 允许在自身头部/尾部形成的矩形区域内寻找中心点
                if cell and cell.cell_type in ["empty", "own_head", "own_tail", "score_boost"]:
                     empty_mask[r, c] = True

        heights = np.zeros(cols, dtype=int) # 存储当前行每个位置向上的连续空单元格高度
        rectangles = [] # 存储找到的矩形信息 (area, center_x, center_y)

        for r in range(rows):
            # 1. 更新当前行的高度数组
            for c in range(cols):
                if empty_mask[r, c]:
                    heights[c] += 1
                else:
                    heights[c] = 0 # 遇到障碍物，高度归零

            # 2. 计算当前高度数组（柱状图）中的最大矩形
            # 使用栈来高效计算，添加哨兵简化边界处理
            stack = [-1] # 栈底哨兵
            heights_with_sentinel = np.append(heights, 0) # 末尾哨兵

            for c, h in enumerate(heights_with_sentinel):
                # 当遇到更短的柱子或末尾哨兵时，处理栈中更高的柱子
                while heights_with_sentinel[stack[-1]] > h:
                    height = heights_with_sentinel[stack.pop()]
                    # 栈顶弹出后，新的栈顶就是左边界（不包含）
                    width = c - stack[-1] - 1

                    if height > 0 and width > 0:
                        area = height * width
                        # 计算矩形的实际坐标范围
                        # 左上角: (stack[-1] + 1, r - height + 1)
                        # 右下角: (c - 1, r)
                        center_x = (stack[-1] + 1 + c - 1) // 2
                        center_y = (r - height + 1 + r) // 2
                        rectangles.append((area, center_x, center_y))

                # 将当前柱子索引压入栈
                stack.append(c)

        # 3. 按面积从大到小排序
        rectangles.sort(reverse=True, key=lambda x: x[0])

        # 4. 过滤掉重复的中心点（面积大的优先）
        unique_centers = []
        seen_centers = set()
        for area, cx, cy in rectangles:
            if (cx, cy) not in seen_centers:
                 unique_centers.append((cx, cy))
                 seen_centers.add((cx, cy))

        return unique_centers
    
    def flood_fill_area_estimate(self, start_node, board, max_cells_to_visit):
        """从起点进行受限的洪水填充/BFS，计算在限制内能访问到的安全空格数量。
        
        Args:
            start_node: 起始节点坐标 (x, y)
            board: 棋盘对象
            max_cells_to_visit: 最大访问单元格数量
            
        Returns:
            可访问的安全空格数量
        """
        if not start_node:
            return 0

        # 定义常量
        MOVES = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 上下左右
        SAFE_TYPES = {"empty", "score_boost", "own_tail"}
        PASSABLE_TYPES = {"empty", "score_boost", "own_tail", "own_head"}

        # 使用maxlen限制队列大小
        q = deque([start_node], maxlen=max_cells_to_visit)
        visited = {start_node}
        area_count = 0
        cells_visited_count = 0

        # 检查起始节点
        start_x, start_y = start_node
        if not (0 <= start_x < board.cols and 0 <= start_y < board.rows):
            return 0

        start_cell = board.cells[start_y][start_x]
        start_type = start_cell.cell_type if start_cell else "wall"
        
        if start_type in SAFE_TYPES:
            area_count = 1
        elif start_type not in ["own_head"]:
            return 0

        while q and cells_visited_count < max_cells_to_visit:
            current_x, current_y = q.popleft()
            cells_visited_count += 1

            # 使用NumPy进行向量化计算
            neighbors = np.array([(current_x + dx, current_y + dy) for dx, dy in MOVES])
            valid_mask = (
                (neighbors[:, 0] >= 0) & (neighbors[:, 0] < board.cols) &
                (neighbors[:, 1] >= 0) & (neighbors[:, 1] < board.rows)
            )

            for nx, ny in neighbors[valid_mask]:
                next_node = (nx, ny)
                if next_node in visited:
                    continue

                cell = board.cells[ny][nx]
                cell_type = cell.cell_type if cell else "wall"

                if cell_type in PASSABLE_TYPES:
                    visited.add(next_node)
                    q.append(next_node)
                    if cell_type in SAFE_TYPES:
                        area_count += 1
                else:
                    visited.add(next_node)

        return area_count
    
    def get_available_directions(self, board, current_pos):
        """获取当前位置四方向中空白格的方向
        
        Args:
            board: 棋盘对象
            current_pos: 当前位置坐标 (x, y)
            
        Returns:
            可用方向列表，例如 ['up', 'right']
        """
        available_directions = []
        x, y = current_pos
        moves = [(0, -1, "up"), (0, 1, "down"), (-1, 0, "left"), (1, 0, "right")]
        for dx, dy, d in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < board.cols and 0 <= ny < board.rows:
                cell = board.cells[ny][nx]
                if cell and cell.cell_type == "empty":
                    available_directions.append(d)
        return available_directions
    
    def get_current_direction(self, board, head_pos):
        """获取蛇当前运动方向
        
        Args:
            board: 棋盘对象
            head_pos: 蛇头位置坐标 (x, y)
            
        Returns:
            方向向量 (dx, dy) 或 None
        """
        if not hasattr(board, 'own_snake'):
            return None
            
        head_x, head_y = head_pos
        current_direction_vector = None
        
        if len(board.own_snake) >= 2:
            prev_x, prev_y = board.own_snake[1]
            current_direction_vector = (head_x - prev_x, head_y - prev_y)
        elif hasattr(board, 'direction') and board.direction:
            dir_map = {
                "up": (0, -1),
                "down": (0, 1),
                "left": (-1, 0),
                "right": (1, 0)
            }
            if board.direction in dir_map:
                current_direction_vector = dir_map[board.direction]
                
        return current_direction_vector