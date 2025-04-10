from collections import deque
import random

from PyQt5.QtCore import QObject, pyqtSignal

class PathFinder(QObject):
    """路径查找器类，用于在网格中寻找路径"""
    # 定义信号
    path_updated = pyqtSignal(object, object)  # 当前路径、目标位置
    
    def __init__(self, grid_height, grid_width, logger=None):
        """
        初始化路径查找器
        :param grid_height: 网格高度
        :param grid_width: 网格宽度
        :param logger: 日志记录器对象，默认为None
        """
        super().__init__()
        self.grid_height = grid_height  # 网格高度
        self.grid_width = grid_width    # 网格宽度
        self.start_pos = None           # 起始位置
        self.current_direction = 'right' # 当前方向
        self.logger = logger            # 日志记录器
        # 高风险区域（敌方蛇头附近1格）和低风险区域（敌方身体附近1格）
        self.high_risk_areas = set()
        self.low_risk_areas = set()
        self.mid_risk_areas = set()

    def update_risk_areas(self, board):
        """
        根据棋盘特殊点更新高风险、低风险、中风险区域
        """
        self.high_risk_areas.clear()
        self.low_risk_areas.clear()
        self.mid_risk_areas.clear()

        # 标记敌方蛇头周围1格为高风险
        if "enemy_head" in board.special_cells:
            for cell in board.special_cells["enemy_head"]:
                x, y = cell.col, cell.row
                for dx, dy in [(0, -1), (0,1),(-1,0),(1,0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        self.high_risk_areas.add((nx, ny))

        # 敌方蛇身附近1格改为中风险
        if "enemy_body" in board.special_cells:
            for cell in board.special_cells["enemy_body"]:
                x, y = cell.col, cell.row
                for dx, dy in [(0, -1), (0,1),(-1,0),(1,0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        self.mid_risk_areas.add((nx, ny))

        # 自己蛇身附近1格设为低风险
        if "own_body" in board.special_cells:
            for cell in board.special_cells["own_body"]:
                x, y = cell.col, cell.row
                for dx, dy in [(0, -1), (0,1),(-1,0),(1,0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        self.low_risk_areas.add((nx, ny))
    
    def find_path_bfs(self, start, target, board, init_path=None, min_path_length=2):
        """
        广度优先搜索（BFS）寻路，带危险等级优先
        :param start: 起点(x, y)，用于首次寻路或忽略已有路径
        :param target: 目标位置(x, y)
        :param board: 棋盘状态
        :param init_path: 已有路径（可选）
        :param min_path_length: 返回路径的最小长度，默认2（含起点+目标）
        """
        if not board or not board.cells:
            return None

        self.update_risk_areas(board)
        
        # 从安全 -> 低风险 -> 高风险，逐层放宽
        for risk_threshold in range(3):
            if init_path and len(init_path) > 0:
                new_start = init_path[-1]
                visited = set(init_path)
                if new_start == target:
                    return init_path if len(init_path) >= min_path_length else None
                queue = deque([(new_start, init_path.copy())])
            else:
                visited = {start}
                queue = deque([(start, [start])])

            found_path = None

            while queue:
                current_pos, path = queue.popleft()
                x, y = current_pos

                if current_pos == target:
                    if len(path) >= min_path_length:
                        found_path = path
                        break
                    else:
                        continue

                moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]

                def risk_level(nx, ny):
                    pos = (nx, ny)
                    if pos in self.high_risk_areas:
                        return 2
                    elif hasattr(self, 'mid_risk_areas') and pos in self.mid_risk_areas:
                        return 1.5  # 中风险介于1~2
                    elif pos in self.low_risk_areas:
                        return 1
                    else:
                        return 0

                moves.sort(key=lambda m: risk_level(x + m[0], y + m[1]))

                for dx, dy in moves:
                    next_x, next_y = x + dx, y + dy
                    next_pos = (next_x, next_y)
                    if 0 <= next_x < self.grid_width and 0 <= next_y < self.grid_height:
                        cell = board.cells[next_y][next_x]
                        cell_type = cell.cell_type if cell else None
                        is_valid = (cell_type in ["empty", "score_boost", "own_head"]) or next_pos == target
                        if not is_valid:
                            continue

                        # 核心：只允许小于等于当前阈值的风险
                        if risk_level(next_x, next_y) > risk_threshold:
                            continue

                        if next_pos not in visited:
                            visited.add(next_pos)
                            queue.append((next_pos, path + [next_pos]))

            if found_path:
                return found_path  # 找到较优路径即返回，避免放宽

        return None  # 全部尝试后都无法抵达

    def find_path_to_score_boost(self, board, direction=None):
        """
        尝试往分数点，再接尾巴，确保安全
        输入：
            board: 当前棋盘
            direction: 当前方向
        输出：
            路径 或 None（表示死路）
        """
        self.current_direction = direction if direction else board.direction
        if "own_head" not in board.special_cells or not board.special_cells["own_head"]:
            return None
        head_cell = board.special_cells["own_head"][0]
        self.start_pos = (head_cell.col, head_cell.row)

        score_boosts = []
        if "score_boost" in board.special_cells:
            for cell in board.special_cells["score_boost"]:
                score_boosts.append((cell.col, cell.row))

        for target in score_boosts:
            path1 = self.find_path_bfs(self.start_pos, target, board)
            if path1:
                # 找到去score的路后，再接着往尾巴找
                if "own_tail" not in board.special_cells or not board.special_cells["own_tail"]:
                    continue  # 无尾巴坐标，换下一个分数点
                tail_cell = board.special_cells["own_tail"][0]
                tail_pos = (tail_cell.col, tail_cell.row)
                path2 = self.find_path_bfs(None, tail_pos, board, init_path=path1, min_path_length=3)
                if path2:
                    return path2
        return None

    def find_random_path(self, board):
        """
        随机选择一个空格并查路径（无目标保命时备用）
        """
        empty_cells = []
        if "empty" in board.special_cells:
            for cell in board.special_cells["empty"]:
                empty_cells.append((cell.col, cell.row))
        if not empty_cells:
            return None
        target = random.choice(empty_cells)
        return self.find_path_bfs(self.start_pos, target, board)

    def update_board_with_enemy_heads(self, board):
        """
        老版-直接在棋盘上将敌方蛇头扩散2格范围内全部设为“enemy_head”
        实际中已不推荐用，而用update_risk_areas

        返回：
            修改后的board对象
        """
        # 将敌蛇蛇头附近五格(即半径2范围)都标记成enemy_head
        enemy_heads = []
        if "enemy_head" in board.special_cells:
            for cell in board.special_cells["enemy_head"]:
                enemy_heads.append((cell.col, cell.row))
        for x0, y0 in enemy_heads:
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    x, y = x0 + dx, y0 + dy
                    if 0 <= x < board.cols and 0 <= y < board.rows:
                        cell = board.cells[y][x]
                        if cell:
                            cell.cell_type = "enemy_head"
        return board

    def find_path_to_tail(self, board):
        """
        计算去往自己尾部的路径，用于防止围死自己
        """
        self.current_direction = board.direction
        if "own_head" not in board.special_cells or not board.special_cells["own_head"]:
            return None
        head_cell = board.special_cells["own_head"][0]
        head_pos = (head_cell.col, head_cell.row)

        # 计算蛇头前方一格作为起点，避免判定蛇身无路
        x, y = head_pos
        if self.current_direction == 'up':
            start_pos = (x, y - 1)
        elif self.current_direction == 'down':
            start_pos = (x, y + 1)
        elif self.current_direction == 'left':
            start_pos = (x - 1, y)
        elif self.current_direction == 'right':
            start_pos = (x + 1, y)
        else:
            start_pos = head_pos

        if not (0 <= start_pos[0] < board.cols and 0 <= start_pos[1] < board.rows):
            return None

        self.start_pos = start_pos

        if "own_tail" not in board.special_cells:
            return None
        tail_cell = board.special_cells["own_tail"][0]
        tail_pos = (tail_cell.col, tail_cell.row)

        return self.find_path_bfs(self.start_pos, tail_pos, board)

    def find_path_to_nearest_empty(self, board):
        """
        寻找离蛇头最近、且可以走的空格；若没有就用随机方向兜底
        """
        self.current_direction = board.direction

        # 定位蛇头
        head_pos = None
        for r in range(board.rows):
            for c in range(board.cols):
                cell = board.cells[r][c]
                if cell and cell.cell_type == "own_head":
                    head_pos = (c, r)
                    break
            if head_pos:
                break
        if not head_pos:
            return None

        dirs = self.get_available_directions(board, head_pos)
        if dirs:
            next_dir = random.choice(dirs)
            x, y = head_pos
            if next_dir == "up":
                return [head_pos, (x, y - 1)]
            elif next_dir == "down":
                return [head_pos, (x, y + 1)]
            elif next_dir == "left":
                return [head_pos, (x - 1, y)]
            elif next_dir == "right":
                return [head_pos, (x + 1, y)]
        return None

    def get_available_directions(self, board, current_pos):
        """
        获取当前位置四方向中空白格的方向

        返回：示例 ['up', 'right']
        """
        available_directions = []
        x, y = current_pos
        moves = [(0, -1, "up"), (0,1,"down"), (-1,0,"left"), (1,0,"right")]
        for dx, dy, d in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < board.cols and 0 <= ny < board.rows:
                cell = board.cells[ny][nx]
                if cell and cell.cell_type == "empty":
                    available_directions.append(d)
        return available_directions

    def find_path_in_order(self, board):
        """
        优先策略顺序寻路（分数道具优先，再找尾巴，再瞎走）

        返回：
            路径 或 None
        """

        path = self.find_path_to_score_boost(board)
        if path:
            return path

        path = self.find_path_to_tail(board)
        if path:
            return path

        path = self.find_path_to_nearest_empty(board)
        if path:
            return path
        return None