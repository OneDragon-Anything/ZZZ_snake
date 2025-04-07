import time
import heapq
import numpy as np
from log import SnakeLogger

class PathCalculator:
    """负责根据地图计算路径的类"""
    def __init__(self, logger=None):
        self.logger = logger
        
        # 路径计算优化相关变量
        self.path_cache = {}  # 路径缓存字典
        self.last_path_calc_time = 0  # 上次路径计算时间
        self.path_calc_interval = 0.1  # 路径计算间隔（秒）
        self.last_board_state = None  # 上次的棋盘状态
        self.last_path = None  # 上次计算的路径
        self.path_calc_count = 0  # 路径计算次数计数器
        
    def set_path_calc_interval(self, interval):
        """设置路径计算间隔"""
        self.path_calc_interval = interval
        self.path_cache = {}  # 清除路径缓存
        self.last_path = None
        
    def calculate_path(self, board_state, start_pos):
        """
        计算从起始位置到目标的最佳路径
        :param board_state: 棋盘状态
        :param start_pos: 起始位置 (x, y)
        :return: 路径坐标列表
        """
        current_time = time.time()
        
        # 检查是否可以使用缓存的路径
        if self.last_path and current_time - self.last_path_calc_time < self.path_calc_interval:
            # 检查蛇头位置是否在路径中
            head_in_path = False
            head_index = -1
            
            # 确保start_pos是蛇头位置
            head_found = False
            for r in range(len(board_state)):
                for c in range(len(board_state[r])):
                    if board_state[r][c] == 'own_head':
                        start_pos = (c, r)  # 使用实际的蛇头位置
                        head_found = True
                        break
                if head_found:
                    break
            
            if head_found:
                for i, pos in enumerate(self.last_path):
                    if pos == start_pos:
                        head_in_path = True
                        head_index = i
                        break
                
                # 如果蛇头在路径中，返回从蛇头开始的剩余路径
                if head_in_path and head_index < len(self.last_path) - 1:
                    return self.last_path[head_index:]
        
        # 增加路径计算计数
        self.path_calc_count += 1
        if self.path_calc_count % 10 == 0 and self.logger:  # 每10次计算输出一次日志
            self.logger.log(f"路径计算次数: {self.path_calc_count}")
        
        # 寻找最近的奖励（食物、加速、加分）
        closest_reward = self.find_closest_reward(board_state, start_pos)
        
        # 如果找不到奖励，尝试找一个安全的空格子
        if not closest_reward:
            closest_reward = self.find_safe_empty_cell(board_state, start_pos)
        
        # 如果仍然找不到目标，返回简单的直线路径
        if not closest_reward:
            return self.calculate_simple_path(board_state, start_pos)
        
        # 使用A*算法计算路径
        path = self.a_star_search(board_state, start_pos, closest_reward)
        
        # 如果找不到路径，尝试简单路径
        if not path:
            path = self.calculate_simple_path(board_state, start_pos)
        
        # 优化路径
        optimized_path = self.optimize_path(path)
        
        # 更新缓存
        self.last_path = optimized_path
        self.last_path_calc_time = current_time
        
        return optimized_path
    
    def find_closest_reward(self, board_state, start_pos):
        """
        寻找最近的奖励
        :param board_state: 棋盘状态
        :param start_pos: 起始位置 (x, y)
        :return: 最近奖励的位置 (x, y) 或 None
        """
        rewards = []
        start_x, start_y = start_pos
        
        # 遍历棋盘寻找奖励
        for y in range(len(board_state)):
            for x in range(len(board_state[y])):
                cell = board_state[y][x]
                if cell in ['speed_boost', 'score_boost']:
                    # 计算曼哈顿距离
                    distance = abs(x - start_x) + abs(y - start_y)
                    rewards.append(((x, y), distance))
        
        # 按距离排序
        rewards.sort(key=lambda x: x[1])
        
        # 返回最近的奖励位置
        return rewards[0][0] if rewards else None
    
    def find_safe_empty_cell(self, board_state, start_pos):
        """
        寻找安全的空格子
        :param board_state: 棋盘状态
        :param start_pos: 起始位置 (x, y)
        :return: 安全空格子的位置 (x, y) 或 None
        """
        empty_cells = []
        start_x, start_y = start_pos
        
        # 遍历棋盘寻找空格子
        for y in range(len(board_state)):
            for x in range(len(board_state[y])):
                if board_state[y][x] == 'empty':
                    # 检查是否安全（不靠近敌人）
                    is_safe = True
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < len(board_state) and 0 <= nx < len(board_state[0]):
                                if board_state[ny][nx] in ['enemy_head', 'enemy_body']:
                                    is_safe = False
                                    break
                        if not is_safe:
                            break
                    
                    if is_safe:
                        # 计算曼哈顿距离
                        distance = abs(x - start_x) + abs(y - start_y)
                        empty_cells.append(((x, y), distance))
        
        # 按距离排序
        empty_cells.sort(key=lambda x: x[1])
        
        # 返回最近的安全空格子位置
        return empty_cells[0][0] if empty_cells else None
    
    def calculate_simple_path(self, board_state, start_pos):
        """
        计算简单路径（直线移动）
        :param board_state: 棋盘状态
        :param start_pos: 起始位置 (x, y)
        :return: 路径坐标列表
        """
        # 初始化路径，包含起始位置
        path = [start_pos]
        x, y = start_pos
        
        # 尝试四个方向，选择第一个安全的方向
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # 上、右、下、左
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # 检查是否在边界内
            if 0 <= ny < len(board_state) and 0 <= nx < len(board_state[0]):
                # 检查是否安全（不是障碍物或敌人）
                if board_state[ny][nx] not in ['obstacle', 'enemy_head', 'enemy_body', 'own_body']:
                    path.append((nx, ny))
                    return path
        
        # 如果没有安全方向，返回只包含起始位置的路径
        return path
    
    def a_star_search(self, board_state, start, goal):
        """
        A*搜索算法
        :param board_state: 棋盘状态
        :param start: 起始位置 (x, y)
        :param goal: 目标位置 (x, y)
        :return: 路径坐标列表
        """
        # 转换坐标格式 (x,y) -> (row,col)
        start_row, start_col = start[1], start[0]
        goal_row, goal_col = goal[1], goal[0]
        
        # 创建缓存键
        cache_key = (start_row, start_col, goal_row, goal_col)
        
        # 检查缓存
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # 定义启发式函数（曼哈顿距离）
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        # 定义邻居函数
        def get_neighbors(row, col):
            neighbors = []
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 右、下、左、上
                nr, nc = row + dr, col + dc
                if 0 <= nr < len(board_state) and 0 <= nc < len(board_state[0]):
                    # 检查是否可通行
                    if board_state[nr][nc] not in ['obstacle', 'enemy_head', 'enemy_body', 'own_body']:
                        neighbors.append((nr, nc))
            return neighbors
        
        # 初始化开放集和关闭集
        open_set = []
        closed_set = set()
        
        # 将起点加入开放集
        heapq.heappush(open_set, (0, 0, (start_row, start_col), [(start_col, start_row)]))
        
        while open_set:
            # 取出f值最小的节点
            f, g, current, path = heapq.heappop(open_set)
            
            # 如果到达目标，返回路径
            if current == (goal_row, goal_col):
                # 转换回 (x,y) 格式并缓存
                result_path = [(col, row) for row, col in [(p[1], p[0]) for p in path]]
                self.path_cache[cache_key] = result_path
                return result_path
            
            # 如果已经访问过，跳过
            if current in closed_set:
                continue
            
            # 标记为已访问
            closed_set.add(current)
            
            # 检查邻居
            for neighbor in get_neighbors(*current):
                if neighbor in closed_set:
                    continue
                
                # 计算新的g值
                new_g = g + 1
                
                # 计算f值
                h = heuristic(neighbor, (goal_row, goal_col))
                f = new_g + h
                
                # 创建新路径
                new_path = list(path)
                new_path.append((neighbor[1], neighbor[0]))  # 转换为 (x,y) 格式
                
                # 加入开放集
                heapq.heappush(open_set, (f, new_g, neighbor, new_path))
        
        # 如果找不到路径，返回None
        return None
    
    def optimize_path(self, path):
        """
        优化路径，去除不必要的拐点
        :param path: 原始路径
        :return: 优化后的路径
        """
        if not path or len(path) <= 2:
            return path
        
        # 初始化优化后的路径，包含起点
        optimized = [path[0]]
        i = 0
        
        # 遍历路径点
        while i < len(path):
            current = path[i]
            
            # 寻找可以直接到达的最远点
            furthest = i
            
            if i + 1 < len(path):
                next_point = path[i + 1]
                
                # 如果是直线移动，尝试找到最远的直线点
                if next_point[0] == current[0] or next_point[1] == current[1]:
                    dx = 1 if next_point[0] > current[0] else (-1 if next_point[0] < current[0] else 0)
                    dy = 1 if next_point[1] > current[1] else (-1 if next_point[1] < current[1] else 0)
                    
                    # 沿着直线方向寻找最远点
                    for j in range(i + 1, len(path)):
                        # 检查是否在同一直线上
                        if dx != 0:
                            if (path[j][0] - current[0]) // dx >= 0:  # 使用整除代替取模
                                furthest = j
                            else:
                                break
                        elif dy != 0:
                            if (path[j][1] - current[1]) // dy >= 0:  # 使用整除代替取模
                                furthest = j
                            else:
                                break
                        else:
                            furthest = j
            
            # 添加最远点到优化路径
            if furthest > i:
                optimized.append(path[furthest])
            
            # 移动到下一个点
            i = furthest + 1
        
        # 确保路径至少包含起点
        if not optimized and path:
            optimized.append(path[0])
        
        return optimized