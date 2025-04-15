import numpy as np
import heapq
from collections import deque
import random


class PathfindingAlgorithm:
    """路径查找算法基类，提供通用方法和接口"""
    
    def __init__(self, grid_height, grid_width, risk_array):
        """初始化路径查找算法
        
        Args:
            grid_height: 网格高度
            grid_width: 网格宽度
            risk_array: 风险数组引用
        """
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.risk_array = risk_array  # 引用外部风险数组
        
    def is_valid_position(self, x, y):
        """检查位置是否在网格范围内"""
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height
    
    def is_passable(self, x, y, board, target=None):
        """检查位置是否可通行
        
        Args:
            x, y: 坐标
            board: 棋盘对象
            target: 目标位置，如果是目标位置则特殊处理
        """
        if not self.is_valid_position(x, y):
            return False
            
        # 如果是目标位置，特殊处理
        if target and (x, y) == target:
            return True
            
        cell = board.cells[y][x]
        cell_type = cell.cell_type if cell else None
        return cell_type in ["empty", "score_boost", "own_head", "own_tail"]
    
    def risk_penalty(self, x, y):
        """获取位置的风险值"""
        if self.is_valid_position(x, y):
            return self.risk_array[y+1, x+1]
        return 0
    
    def find_path(self, start, target, board, init_path=None, min_path_length=2):
        """查找路径的抽象方法，子类必须实现"""
        raise NotImplementedError("子类必须实现find_path方法")


class AStarAlgorithm(PathfindingAlgorithm):
    """A*寻路算法实现"""
    
    def heuristic(self, a, b):
        """启发式函数，计算两点间的曼哈顿距离"""
        return 1.2 * (abs(a[0] - b[0]) + abs(a[1] - b[1]))
    
    def find_path(self, start, target, board, init_path=None, min_path_length=2):
        """使用A*算法寻找路径
        
        Args:
            start: 起点坐标 (x, y)
            target: 目标坐标 (x, y)
            board: 棋盘对象
            init_path: 初始路径，用于断点续寻
            min_path_length: 最小路径长度
            
        Returns:
            路径列表或None
        """
        # 判断起点
        if init_path and len(init_path) > 0:
            new_start = init_path[-1]

            # 检查续寻起点是否可通行
            if new_start != target:  # 如果起点不是目标
                x, y = new_start
                if not self.is_valid_position(x, y):
                    return None

                if not self.is_passable(x, y, board, target):
                    return None
        elif start:
            new_start = start
            init_path = []
        else:
            return None

        # 检查目标点是否可达
        tx, ty = target
        if not self.is_valid_position(tx, ty):
            return None

        target_cell = board.cells[ty][tx]
        target_type = target_cell.cell_type if target_cell else None
        if target_type not in ["empty", "score_boost", "own_head", "own_tail"]:
            return None

        # 使用NumPy数组存储g_score，提高访问效率
        g_score_array = np.full((self.grid_height, self.grid_width), np.inf, dtype=np.float32)
        open_set = []
        count = 0
        came_from = dict()
        # 记录每个节点的来源方向
        node_directions = {}
        
        # 初始化已有路径中每个点的累积代价
        total_cost = 0
        prev = None

        # 处理初始路径
        for idx, pt in enumerate(init_path):
            x, y = pt
            if idx == 0:
                g_score_array[y, x] = 0
            else:
                penalty = self.risk_penalty(x, y)

                # 计算方向变化并增加拐弯代价
                if idx >= 2:
                    prev_prev = init_path[idx - 2]
                    prev = init_path[idx - 1]
                    curr_dir = (x - prev[0], y - prev[1])
                    prev_dir = (prev[0] - prev_prev[0], prev[1] - prev_prev[1])
                    if curr_dir != prev_dir:  # 方向改变，增加拐弯代价
                        penalty += 2  # 拐弯额外代价
                    # 记录方向
                    node_directions[pt] = curr_dir

                g_score_array[y, x] = total_cost + 1 + penalty
                came_from[pt] = prev
                total_cost = g_score_array[y, x]
            prev = pt

        # 将续算起点加入开放列表，并初始化其g_score
        x, y = new_start
        g_score_array[y, x] = total_cost  # 确保起点有g_score
        heapq.heappush(
            open_set, (total_cost + self.heuristic(new_start, target), count, new_start)
        )
        count += 1

        # 添加随机性，避免总是走相同路径
        random_factor = 0.1  # 定义随机因子变量

        # 记录找到的最佳路径
        best_path = None

        # 记录每个节点的来源方向
        if len(init_path) >= 2:
            for i in range(1, len(init_path)):
                prev = init_path[i - 1]
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

            # 获取当前节点的g_score
            cx, cy = current
            if g_score_array[cy, cx] == np.inf:
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
                    # 确保不会出现路径指回起点的情况
                    if len(path) > 1 and path[0] == init_path[-1]:
                        result.extend(init_path[:-1])  # 不包括最后一个点，因为它是新路径的起点
                        result.extend(path)
                    else:
                        # 如果新路径不是从初始路径的最后一个点开始，可能是找到了另一条路径
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
            # 使用NumPy数组存储移动方向
            moves = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])
            # 随机打乱顺序
            np.random.shuffle(moves)
            
            # 计算所有邻居位置
            neighbors = np.array([(x, y)]) + moves
            
            # 批量检查边界条件
            valid_mask = (neighbors[:, 0] >= 0) & (neighbors[:, 0] < self.grid_width) & \
                         (neighbors[:, 1] >= 0) & (neighbors[:, 1] < self.grid_height)
            
            for i, valid in enumerate(valid_mask):
                if not valid:
                    continue
                    
                nx, ny = neighbors[i]
                neighbor = (nx, ny)
                
                if not self.is_passable(nx, ny, board, target):
                    continue

                penalty = self.risk_penalty(nx, ny)

                # 计算拐弯代价
                curr_dir = tuple(moves[i])
                if current in node_directions:
                    prev_dir = node_directions[current]
                    if curr_dir != prev_dir:  # 方向改变，增加拐弯代价
                        penalty += 2  # 拐弯额外代价

                tentative_g = g_score_array[cy, cx] + 1 + penalty

                # 使用NumPy数组检查g_score
                if g_score_array[ny, nx] == np.inf or tentative_g < g_score_array[ny, nx]:
                    g_score_array[ny, nx] = tentative_g
                    came_from[neighbor] = current
                    # 记录到达neighbor的方向
                    node_directions[neighbor] = curr_dir

                    # 添加微小随机因子，增加路径多样性
                    random_value = random.random() * random_factor
                    priority = tentative_g + self.heuristic(neighbor, target) + random_value

                    count += 1
                    heapq.heappush(open_set, (priority, count, neighbor))

        # 如果找到了路径但长度不够，返回最长的那个
        if best_path is not None:
            return best_path

        # 尝试放宽条件，允许更短的路径
        if min_path_length > 2:
            return self.find_path(start, target, board, init_path, 2)

        return None


class BFSAlgorithm(PathfindingAlgorithm):
    """广度优先搜索算法实现"""
    
    def find_path(self, start, target, board, init_path=None, min_path_length=2):
        """使用BFS算法寻找路径
        
        Args:
            start: 起点坐标 (x, y)
            target: 目标坐标 (x, y)
            board: 棋盘对象
            init_path: 初始路径，用于断点续寻
            min_path_length: 最小路径长度
            
        Returns:
            路径列表或None
        """
        if not board or not board.cells:
            return None

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

                # 使用NumPy数组存储移动方向
                moves = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])

                # 计算所有移动的优先级
                move_priorities = []
                for i, (dx, dy) in enumerate(moves):
                    same_dir = (dir_vec is not None) and ((dx, dy) == dir_vec)
                    move_priorities.append((0 if same_dir else 1, self.risk_penalty(x + dx, y + dy), i))
                
                # 按优先级排序移动方向
                move_priorities.sort()
                moves = moves[[p[2] for p in move_priorities]]

                if current_pos == target:
                    if len(path) >= min_path_length:
                        found_path = path
                        break
                    else:
                        continue

                # 预计算所有邻居位置
                next_positions = np.array([current_pos]) + moves
                
                # 批量检查边界条件
                valid_mask = (next_positions[:, 0] >= 0) & (next_positions[:, 0] < self.grid_width) & \
                             (next_positions[:, 1] >= 0) & (next_positions[:, 1] < self.grid_height)
                
                for i, valid in enumerate(valid_mask):
                    if not valid:
                        continue
                        
                    next_x, next_y = next_positions[i]
                    next_pos = (next_x, next_y)
                    
                    # 检查单元格类型
                    if not self.is_passable(next_x, next_y, board, target):
                        continue

                    # 检查风险等级
                    if self.risk_penalty(next_x, next_y) > risk_threshold:
                        continue

                    # 检查是否已访问
                    if next_pos not in visited:
                        visited.add(next_pos)
                        queue.append((next_pos, path + [next_pos]))

            if found_path:
                return found_path  # 找到较优路径即返回，避免放宽

        return None  # 全部尝试后都无法抵达