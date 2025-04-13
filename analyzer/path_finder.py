from collections import deque
import random
import heapq
import time
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
        # 改为字典存储每个格子的风险分数
        self.risk_scores = {}

        # === 标记地图边缘为高风险 ===
        for x in range(self.grid_width):
            self._add_risk_score((x, 0), 9)
            self._add_risk_score((x, self.grid_height - 1), 9)
        for y in range(self.grid_height):
            self._add_risk_score((0, y), 9)
            self._add_risk_score((self.grid_width - 1, y), 9)

        # === 标记地图边缘外一格为中风险 ===
        for x in range(-1, self.grid_width + 1):
            self._add_risk_score((x, -1), 3)
            self._add_risk_score((x, self.grid_height), 3)
        for y in range(-1, self.grid_height + 1):
            self._add_risk_score((-1, y), 3)
            self._add_risk_score((self.grid_width, y), 3)

        # 标记敌方蛇头周围1格为高风险
        for key in ['enemy_head', 'mine', 'unknow']:
            if key in board.special_cells:
                for cell in board.special_cells[key]:
                    x, y = cell.col, cell.row
                    for dx, dy in [(0, -1), (0,1),(-1,0),(1,0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                            self._add_risk_score((nx, ny), 9)

        # 敌方蛇身附近1格为中风险
        if "enemy_body" in board.special_cells:
            for cell in board.special_cells["enemy_body"]:
                x, y = cell.col, cell.row
                for dx, dy in [(0, -1), (0,1),(-1,0),(1,0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        self._add_risk_score((nx, ny), 3)

        # 自己蛇身附近1格为低风险
        for key in ['own_body', 'greed_speed']:
            if key in board.special_cells:
                for cell in board.special_cells[key]:
                    x, y = cell.col, cell.row
                    for dx, dy in [(0, -1), (0,1),(-1,0),(1,0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                            self._add_risk_score((nx, ny), 1)

    def _add_risk_score(self, pos, score):
        """辅助方法：累加风险分数"""
        if pos in self.risk_scores:
            self.risk_scores[pos] += score
        else:
            self.risk_scores[pos] = score
    

    def find_path(self, start, target, board, init_path=None, min_path_length=2, method='A'):
        """
        统一寻路接口：支持A*和BFS
        参数：
            method:  'A' 用A*， 'B'用BFS，默认'B'
        返回：
            路径list 或 None
        """
        t0 = time.time()
        if method == 'A':
            path = self.find_path_astar(start, target, board, init_path=init_path, min_path_length=min_path_length)
        else:
            path = self.find_path_bfs(start, target, board, init_path=init_path, min_path_length=min_path_length)
        t1 = time.time()
        
        return path


    def find_path_astar(self, start, target, board, init_path=None, min_path_length=2):
        """
        使用A*算法路径寻路，可以断点续寻
        返回拼接路径列表[start,...,target]，无法到达时返回None
        """

        self.update_risk_areas(board)

        # 判断起点
        if init_path and len(init_path) > 0:
            new_start = init_path[-1]
            
            # 检查续寻起点是否可通行
            if new_start != target:  # 如果起点不是目标
                x, y = new_start
                if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
                    return None
                
                cell = board.cells[y][x]
                cell_type = cell.cell_type if cell else None
                if cell_type not in ["empty", "score_boost", "own_head"] and new_start != target:
                    return None
        elif start:
            new_start = start
            init_path = []
        else:
            return None

        # 检查目标点是否可达
        tx, ty = target
        if not (0 <= tx < self.grid_width and 0 <= ty < self.grid_height):
            return None
            
        target_cell = board.cells[ty][tx]
        target_type = target_cell.cell_type if target_cell else None
        if target_type not in ["empty", "score_boost", "own_head", "own_tail"]:
            return None

        # 定义风险权重
        def risk_penalty(x, y):
            return self.risk_scores.get((x, y), 0)

        def heuristic(a, b):
            # 增加启发式函数权重，使路径更倾向于直线
            return 1.2 * (abs(a[0]-b[0]) + abs(a[1]-b[1]))

        open_set = []
        count = 0
        came_from = dict()
        g_score = dict()
        # 修改：不再使用visited集合来排除已访问节点，而是通过g_score判断
        # visited = set()  # 记录已访问节点

        # 初始化已有路径中每个点的累积代价
        total_cost = 0
        prev = None
        
        # 记录每个节点的来源方向
        node_directions = {}
        
        # 处理初始路径
        for idx, pt in enumerate(init_path):
            # 不再将初始路径点加入visited集合
            # visited.add(pt)
            if idx == 0:
                g_score[pt] = 0
            else:
                penalty = risk_penalty(pt[0], pt[1])
                
                # 计算方向变化并增加拐弯代价
                if idx >= 2:
                    prev_prev = init_path[idx-2]
                    prev = init_path[idx-1]
                    curr_dir = (pt[0] - prev[0], pt[1] - prev[1])
                    prev_dir = (prev[0] - prev_prev[0], prev[1] - prev_prev[1])
                    if curr_dir != prev_dir:  # 方向改变，增加拐弯代价
                        penalty += 2  # 拐弯额外代价
                    # 记录方向
                    node_directions[pt] = curr_dir
                
                g_score[pt] = total_cost + 1 + penalty
                came_from[pt] = prev
                total_cost = g_score[pt]
            prev = pt

        # 将续算起点加入开放列表，并初始化其g_score
        g_score[new_start] = total_cost  # 修复：确保起点有g_score
        heapq.heappush(open_set, (total_cost + heuristic(new_start, target), count, new_start))
        count += 1
        
        # 添加随机性，避免总是走相同路径
        random_factor = 0.1  # 定义随机因子变量
        
        # 记录找到的最佳路径
        best_path = None
        
        # 记录每个节点的来源方向
        if len(init_path) >= 2:
            for i in range(1, len(init_path)):
                prev = init_path[i-1]
                curr = init_path[i]
                node_directions[curr] = (curr[0] - prev[0], curr[1] - prev[1])
        
        # 记录搜索状态
        nodes_explored = 0
        max_iterations = 1000  # 防止无限循环
        
        # 使用集合记录已处理的节点，避免重复处理
        processed = set()
        
        while open_set and nodes_explored < max_iterations:
            nodes_explored += 1
            f_val, _, current = heapq.heappop(open_set)

            # 防止 KeyError: 如果节点没有初始化g_score则跳过
            if current not in g_score:
                continue
                
            # 如果当前节点已处理过，跳过
            if current in processed:
                continue
                
            # 标记当前节点为已处理
            processed.add(current)

            if current == target:
                # 重建新路径
                path = [current]
                temp = current
                while temp in came_from:
                    temp = came_from[temp]
                    path.append(temp)
                path.reverse()
                
                result = []
                if init_path:
                    # 修复：确保不会出现路径指回起点的情况
                    # 检查新路径的第一个点是否就是初始路径的最后一个点
                    if len(path) > 1 and path[0] == init_path[-1]:
                        result.extend(init_path[:-1])  # 不包括最后一个点，因为它是新路径的起点
                        result.extend(path)
                    else:
                        # 如果新路径不是从初始路径的最后一个点开始，可能是找到了另一条路径
                        # 这种情况下，我们只返回新路径，不拼接

                        result = path
                else:
                    result = path
                
                
                # 保存找到的路径，但不立即返回
                if len(result) >= min_path_length:
                    # 找到符合长度要求的路径，立即返回
                    return result
                elif best_path is None or len(result) > len(best_path):
                    # 保存最长的路径，即使不满足最小长度要求
                    best_path = result
                    
                # 继续搜索，看是否能找到更长的路径
                continue

            x, y = current
            # 随机打乱邻居顺序，增加路径多样性
            moves = [(-1,0),(1,0),(0,-1),(0,1)]
            random.shuffle(moves)
            
            for dx, dy in moves:
                nx, ny = x+dx, y+dy
                neighbor = (nx, ny)
                if not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                    continue

                cell = board.cells[ny][nx]
                cell_type = cell.cell_type if cell else None
                is_passable = (cell_type in ["empty", "score_boost", "own_head"]) or neighbor == target
                if not is_passable:
                    continue

                penalty = risk_penalty(nx, ny)
                
                # 计算拐弯代价
                curr_dir = (dx, dy)
                if current in node_directions:
                    prev_dir = node_directions[current]
                    if curr_dir != prev_dir:  # 方向改变，增加拐弯代价
                        penalty += 2  # 拐弯额外代价
                
                tentative_g = g_score[current] + 1 + penalty

                # 只检查g_score，不再使用visited集合来判断是否访问过
                if (neighbor not in g_score) or (tentative_g < g_score[neighbor]):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    # 记录到达neighbor的方向
                    node_directions[neighbor] = curr_dir
                    
                    # 添加微小随机因子，增加路径多样性
                    random_value = random.random() * random_factor
                    priority = tentative_g + heuristic(neighbor, target) + random_value
                    
                    count += 1
                    heapq.heappush(open_set, (priority, count, neighbor))

            
        # 如果找到了路径但长度不够，返回最长的那个
        if best_path is not None:
            return best_path
            
        # 尝试放宽条件，允许更短的路径
        if min_path_length > 2:

            return self.find_path_astar(start, target, board, init_path, 2)
            

        return None  # 找不到


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

                # 计算当前方向：
                if len(path) >= 2:
                    prev_x, prev_y = path[-2]
                    dir_vec = (x - prev_x, y - prev_y)
                else:
                    dir_vec = None

                moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]

                def risk_level(nx, ny):
                    return self.risk_scores.get((x, y), 0)

                def move_priority(m):
                    dx, dy = m
                    same_dir = (dir_vec is not None) and ((dx, dy) == dir_vec)
                    return (0 if same_dir else 1, risk_level(x + dx, y + dy))

                moves.sort(key=move_priority)

                if current_pos == target:
                    if len(path) >= min_path_length:
                        found_path = path
                        break
                    else:
                        continue

                for dx, dy in moves:
                    next_x, next_y = x + dx, y + dy
                    next_pos = (next_x, next_y)
                    if 0 <= next_x < self.grid_width and 0 <= next_y < self.grid_height:
                        cell = board.cells[next_y][next_x]
                        cell_type = cell.cell_type if cell else None
                        is_valid = (cell_type in ["empty", "score_boost", "own_head"]) or next_pos == target
                        if not is_valid:
                            continue

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
        尝试往分数点，再接安全区域，确保安全
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
                x, y = cell.col, cell.row
                is_safe = True
                
                # 检查周围3格范围
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        nx, ny = x + dx, y + dy
                        # 如果超出边界，直接判定为不安全
                        if not (0 <= nx < board.cols and 0 <= ny < board.rows):
                            is_safe = False
                            break
                        # 检查格子类型
                        cell_type = board.cells[ny][nx].cell_type
                        if cell_type not in ["empty", "own_head", "own_body", "own_tail", "score_boost"]:
                            is_safe = False
                            break
                    if not is_safe:
                        break
                
                if is_safe:
                    score_boosts.append((x, y))
                elif self.logger:
                    self.logger.debug(f"[寻路] 分数点({x},{y})周围3格内存在危险或超出边界，跳过")
        
        
        # 没有分数点时直接返回None
        if not score_boosts:
            return None

        # 尝试两种寻路方法，增加成功率
        for method in ['A']:
            for target in score_boosts:
                # 确保目标点是有效的
                if not (0 <= target[0] < board.cols and 0 <= target[1] < board.rows):
                    continue
                    
                path1 = self.find_path(self.start_pos, target, board, method=method, min_path_length=3)
                if path1:
                    if path1[-1] != target:
                        continue
                        
                    # 找到去score的路后，尝试接上安全区域
                    rect = self.find_largest_empty_rectangle(board)
                    x0, y0, w, h = rect
                    if w > 0 and h > 0:
                        safe_target = (x0 + w // 2, y0 + h // 2)
                    else:
                        safe_target = self.find_furthest_from_enemy_heads_bfs(board)
                    
                    # 如果有安全点，考虑拼接逃生路径
                    if safe_target:
                        path2 = self.find_path(None, safe_target, board, init_path=path1, min_path_length=5, method=method)
                        if path2:
                            return path2
                        else:
                            continue
                    else:
                        continue

        return None

    def find_safe_path(self, board):
        """
        智能逃生策略：
        先尝试最大空白矩形的中心点逃跑，
        再退而求其次寻找距离敌方蛇头最远的空白格
        """
        # 1. 最大空白矩形 中心点
        x, y, w, h = self.find_largest_empty_rectangle(board)
        if w > 0 and h > 0:
            center_x = x + w // 2
            center_y = y + h // 2
            path = self.find_path(self.start_pos, (center_x, center_y), board, min_path_length=3, method='B')
            if path:
                return path

        # 2. 距离敌头最远的点
        target = self.find_furthest_from_enemy_heads_bfs(board)
        if target:
            path = self.find_path(self.start_pos, target, board, min_path_length=3, method='B')
            if path:
                return path

        # 3. 全失败，放弃
        return None

    def find_path_to_tail(self, board):
        """
        计算去往自己尾部的路径，用于防止围死自己
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

        path = self.find_path(head_pos, tail_pos, board, min_path_length=5, method='A')
        if path:
            return path

        path = self.find_path(head_pos, tail_pos, board, min_path_length=3, method='A')
        if path:
            return path

        return None

    def find_path_to_nearest_empty(self, board):
        """
        在必死情况下快速寻找任意可行路径
        """
        self.current_direction = board.direction
        
        # 快速定位蛇头
        if "own_head" not in board.special_cells or not board.special_cells["own_head"]:
            return None
        head_cell = board.special_cells["own_head"][0]
        head_pos = (head_cell.col, head_cell.row)

        def dfs(pos, depth, visited):
            if depth >= 5:  # 找到长度>=5的路径就返回
                return visited + [pos]
            
            x, y = pos
            # 直接使用固定顺序，不再考虑风险
            moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)
                
                # 只检查基本可行性
                if (next_pos in visited or 
                    not (0 <= nx < board.cols and 0 <= ny < board.rows)):
                    continue
                    
                cell = board.cells[ny][nx]
                if not cell or cell.cell_type not in ["empty", "score_boost"]:
                    continue
                
                result = dfs(next_pos, depth + 1, visited + [pos])
                if result:
                    return result
            
            return None

        # 直接深搜，不再迭代加深
        return dfs(head_pos, 0, [])

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

    def find_largest_empty_rectangle(self, board):
        """
        动态规划+单调栈，寻找最大空白矩形
        返回：(左上角x, 左上角y, 宽度w, 高度h)
        """
        rows, cols = board.rows, board.cols
        heights = [[0]*cols for _ in range(rows)]
        max_area = 0
        max_rect = (0,0,0,0)

        # 计算每行的高度图
        for y in range(rows):
            for x in range(cols):
                cell = board.cells[y][x]
                if cell and cell.cell_type == "empty":
                    heights[y][x] = 1 + (heights[y-1][x] if y > 0 else 0)
                else:
                    heights[y][x] = 0

        # 对每一行做直方图最大矩形
        for y in range(rows):
            stack = []
            hist = heights[y]+[0]  # 哨兵简化代码
            for i in range(cols+1):
                while stack and hist[i] < hist[stack[-1]]:
                    top_idx = stack.pop()
                    h = hist[top_idx]
                    w = i if not stack else i - stack[-1] - 1
                    area = h * w
                    if area > max_area:
                        max_area = area
                        # 计算矩形左上坐标位置
                        left = stack[-1]+1 if stack else 0
                        top = y - h +1
                        max_rect = (left, top, w, h)
                stack.append(i)
        return max_rect  # (x, y, w, h)

    def find_furthest_from_enemy_heads_bfs(self, board):
        """
        BFS从所有敌人头开始扩散，找到最远的空白格子
        返回：(x,y)
        """
        rows, cols = board.rows, board.cols
        dist_map = [[-1 for _ in range(cols)] for _ in range(rows)]
        from collections import deque

        queue = deque()
        # 初始化：敌头集合
        for key in ['enemy_head', 'enemy2_head', 'enemy3_head']:
            heads = board.special_cells.get(key, [])
            for cell in heads:
                x, y = cell.col, cell.row
                dist_map[y][x] = 0
                queue.append((x,y))
        
        # BFS扩散距离
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if 0<=nx<cols and 0<=ny<rows:
                    cell = board.cells[ny][nx]
                    if not cell or cell.cell_type != "empty":
                        continue
                    if dist_map[ny][nx] == -1:
                        dist_map[ny][nx] = dist_map[y][x] +1
                        queue.append((nx, ny))

        # 找距离最大的空格
        max_dist = -1
        result = None
        for y in range(rows):
            for x in range(cols):
                cell = board.cells[y][x]
                if not cell or cell.cell_type != "empty":
                    continue
                if dist_map[y][x] > max_dist:
                    max_dist = dist_map[y][x]
                    result = (x,y)
        return result

    def find_path_in_order(self, board):
        """
        优先策略顺序寻路（分数道具优先，再找尾巴，再瞎走）

        返回：
            路径 或 None
        """
        path = self.find_path_to_score_boost(board)
        if path:
            return path

        path = self.find_safe_path(board)
        if path:
            return path

        path = self.find_path_to_tail(board)
        if path:
            return path

        path = self.find_path_to_nearest_empty(board)
        if path:
            return path
        return None