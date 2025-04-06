import numpy as np
import heapq
import random
from collections import deque

class MinMaxSnakeAI:
    """
    使用MinMax算法和蒙特卡洛树搜索的贪吃蛇AI决策系统
    结合了A*寻路、Voronoi区域控制和敌方路径预测
    """
    def __init__(self, logger=None):
        self.logger = logger
        self.directions = {'w': (-1, 0), 's': (1, 0), 'a': (0, -1), 'd': (0, 1)}
        self.opposite_directions = {'w': 's', 's': 'w', 'a': 'd', 'd': 'a'}
        self.last_direction = None
        self.danger_weight = 10  # 危险区域权重
        self.food_weight = 5     # 食物吸引权重
        self.space_weight = 2    # 空间权重
        self.edge_buffer = 2     # 边缘缓冲区
        self.search_depth = 3    # MinMax搜索深度
        self.monte_carlo_simulations = 20  # 蒙特卡洛模拟次数
        
    def log(self, message):
        """记录日志信息"""
        if self.logger:
            self.logger.log(message)
    
    def a_star_search(self, board_state, start, end):
        """
        使用A*算法寻找从起点到终点的最佳路径
        比BFS更高效，考虑路径代价和启发式估计
        """
        rows = len(board_state)
        cols = len(board_state[0])
        
        # 优先队列，按f值（g+h）排序
        open_set = []
        # 元素格式: (f, g, 位置, 路径)
        heapq.heappush(open_set, (0, 0, start, [start]))
        
        # 已访问节点集合
        closed_set = set()
        
        while open_set:
            f, g, current, path = heapq.heappop(open_set)
            
            # 如果到达目标
            if current == end:
                return path
            
            # 如果已经访问过
            if current in closed_set:
                continue
                
            # 标记为已访问
            closed_set.add(current)
            
            # 检查四个方向
            r, c = current
            for dr, dc in self.directions.values():
                nr, nc = r + dr, c + dc
                
                # 检查是否在边界内且不是障碍物
                if 0 <= nr < rows and 0 <= nc < cols and board_state[nr][nc] not in ["own_body", "enemy_body"]:
                    neighbor = (nr, nc)
                    
                    # 如果已访问过，跳过
                    if neighbor in closed_set:
                        continue
                    
                    # 计算新的g值（从起点到当前点的代价）
                    new_g = g + 1
                    
                    # 计算h值（启发式函数，曼哈顿距离）
                    h = abs(nr - end[0]) + abs(nc - end[1])
                    
                    # 计算f值
                    f = new_g + h
                    
                    # 增加靠近边缘或敌人的代价
                    is_edge = (nr < self.edge_buffer or nr >= rows - self.edge_buffer or 
                              nc < self.edge_buffer or nc >= cols - self.edge_buffer)
                    
                    is_near_enemy = False
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            if abs(i) + abs(j) <= 1:  # 上下左右和当前格子
                                er, ec = nr + i, nc + j
                                if 0 <= er < rows and 0 <= ec < cols and board_state[er][ec] in ["enemy_head", "enemy_body"]:
                                    is_near_enemy = True
                                    break
                        if is_near_enemy:
                            break
                    
                    # 增加代价
                    if is_edge:
                        f += 3
                    if is_near_enemy:
                        f += 5
                    
                    # 创建新路径
                    new_path = list(path)
                    new_path.append(neighbor)
                    
                    # 添加到开放集合
                    heapq.heappush(open_set, (f, new_g, neighbor, new_path))
        
        # 如果没有找到路径
        return None
    
    def calculate_voronoi(self, board_state, own_head, enemy_heads):
        """
        计算Voronoi图，评估每个格子的控制权
        返回一个二维数组，表示每个格子的控制权（正值表示我方控制，负值表示敌方控制）
        """
        rows = len(board_state)
        cols = len(board_state[0])
        
        # 初始化控制权图
        control_map = np.zeros((rows, cols))
        
        # 对每个空格计算到各蛇头的距离
        for r in range(rows):
            for c in range(cols):
                if board_state[r][c] not in ["own_body", "enemy_body", "own_head", "enemy_head"]:
                    # 计算到我方蛇头的距离
                    own_distance = abs(r - own_head[0]) + abs(c - own_head[1])
                    
                    # 计算到敌方蛇头的最小距离
                    enemy_distance = float('inf')
                    for enemy_r, enemy_c in enemy_heads:
                        dist = abs(r - enemy_r) + abs(c - enemy_c)
                        enemy_distance = min(enemy_distance, dist)
                    
                    # 如果我方距离更近，则为正值；否则为负值
                    if own_distance < enemy_distance:
                        control_map[r][c] = enemy_distance - own_distance
                    elif enemy_distance < own_distance:
                        control_map[r][c] = -(own_distance - enemy_distance)
                    # 距离相等时为0
        
        return control_map
    
    def predict_enemy_moves(self, board_state, enemy_heads):
        """
        预测敌方蛇的可能移动方向
        返回每个敌方蛇头可能的下一步位置列表
        """
        rows = len(board_state)
        cols = len(board_state[0])
        
        predicted_positions = []
        
        for enemy_r, enemy_c in enemy_heads:
            possible_moves = []
            
            # 检查四个方向
            for dr, dc in self.directions.values():
                nr, nc = enemy_r + dr, enemy_c + dc
                
                # 检查是否在边界内且不是障碍物
                if 0 <= nr < rows and 0 <= nc < cols and board_state[nr][nc] not in ["own_body", "enemy_body", "enemy_head"]:
                    # 评估移动的优先级（简单启发式）
                    priority = 0
                    
                    # 检查是否有食物
                    if board_state[nr][nc] in ["score_boost", "speed_boost"]:
                        priority += 3
                    
                    # 避免边缘
                    if nr < self.edge_buffer or nr >= rows - self.edge_buffer or nc < self.edge_buffer or nc >= cols - self.edge_buffer:
                        priority -= 2
                    
                    possible_moves.append(((nr, nc), priority))
            
            # 按优先级排序
            possible_moves.sort(key=lambda x: x[1], reverse=True)
            
            # 添加可能的位置（只取前两个最可能的）
            if possible_moves:
                predicted_positions.extend([pos for pos, _ in possible_moves[:2]])
        
        return predicted_positions
    
    def evaluate_move_safety(self, board_state, head_pos, direction):
        """
        评估移动的安全性，考虑死路和被围困的风险
        返回安全评分（越高越安全）
        """
        rows = len(board_state)
        cols = len(board_state[0])
        r, c = head_pos
        dr, dc = self.directions[direction]
        nr, nc = r + dr, c + dc
        
        # 检查是否在边界内
        if not (0 <= nr < rows and 0 <= nc < cols):
            return -float('inf')  # 出界，极不安全
        
        # 检查是否撞到障碍物
        if board_state[nr][nc] in ["own_body", "enemy_body", "enemy_head"]:
            return -float('inf')  # 撞到障碍物，极不安全
        
        # 使用洪水填充算法计算可用空间
        visited = set()
        queue = deque([(nr, nc)])
        visited.add((nr, nc))
        
        while queue:
            curr_r, curr_c = queue.popleft()
            
            for move_dr, move_dc in self.directions.values():
                next_r, next_c = curr_r + move_dr, curr_c + move_dc
                
                if (0 <= next_r < rows and 0 <= next_c < cols and 
                    board_state[next_r][next_c] not in ["own_body", "enemy_body", "enemy_head"] and 
                    (next_r, next_c) not in visited):
                    visited.add((next_r, next_c))
                    queue.append((next_r, next_c))
        
        # 可用空间越大越安全
        safety_score = len(visited)
        
        # 考虑边缘因素
        if nr < self.edge_buffer or nr >= rows - self.edge_buffer or nc < self.edge_buffer or nc >= cols - self.edge_buffer:
            safety_score -= 5
        
        # 考虑敌方蛇的威胁
        for i in range(-2, 3):
            for j in range(-2, 3):
                check_r, check_c = nr + i, nc + j
                if 0 <= check_r < rows and 0 <= check_c < cols and board_state[check_r][check_c] == "enemy_head":
                    # 敌方蛇头在附近，降低安全性
                    distance = abs(i) + abs(j)
                    safety_score -= (3 - distance) * 10  # 距离越近威胁越大
        
        return safety_score
    
    def minmax_evaluate(self, board_state, own_head, enemy_heads, depth):
        """
        MinMax评估函数，评估当前局面的得分
        """
        # 基本评分
        score = 0
        
        # 如果到达搜索深度限制，使用启发式评估
        if depth == 0:
            # 计算控制区域
            if enemy_heads:
                control_map = self.calculate_voronoi(board_state, own_head, enemy_heads)
                # 计算控制区域总分
                control_score = np.sum(control_map)
                score += control_score * 0.5
            
            # 评估安全性
            rows = len(board_state)
            cols = len(board_state[0])
            r, c = own_head
            
            # 计算可用空间
            available_space = 0
            visited = set([own_head])
            queue = deque([own_head])
            
            while queue:
                curr_r, curr_c = queue.popleft()
                
                for dr, dc in self.directions.values():
                    nr, nc = curr_r + dr, curr_c + dc
                    
                    if (0 <= nr < rows and 0 <= nc < cols and 
                        board_state[nr][nc] not in ["own_body", "enemy_body", "enemy_head"] and 
                        (nr, nc) not in visited):
                        visited.add((nr, nc))
                        queue.append((nr, nc))
                        available_space += 1
            
            score += available_space * 0.3
            
            # 评估与奖励的距离
            rewards = []
            for r in range(rows):
                for c in range(cols):
                    if board_state[r][c] in ["score_boost", "speed_boost"]:
                        rewards.append((r, c))
            
            if rewards:
                min_distance = float('inf')
                for reward_r, reward_c in rewards:
                    distance = abs(own_head[0] - reward_r) + abs(own_head[1] - reward_c)
                    min_distance = min(min_distance, distance)
                
                # 距离越近分数越高
                score += 50 / (min_distance + 1) * self.food_weight
            
            # 评估与敌方蛇的距离
            for enemy_r, enemy_c in enemy_heads:
                distance = abs(own_head[0] - enemy_r) + abs(own_head[1] - enemy_c)
                if distance <= 1:  # 相邻，危险
                    score -= 30
                elif distance <= 3:  # 较近，有风险
                    score -= 10
        
        return score
    
    def minmax(self, board_state, own_head, enemy_heads, depth, is_max):
        """
        MinMax算法实现，考虑我方和敌方的最优决策
        """
        # 如果到达搜索深度限制或游戏结束，返回评估分数
        if depth == 0:
            return self.minmax_evaluate(board_state, own_head, enemy_heads, depth)
        
        rows = len(board_state)
        cols = len(board_state[0])
        
        if is_max:  # 我方回合，寻找最大值
            max_score = -float('inf')
            
            # 尝试每个可能的移动
            for direction, (dr, dc) in self.directions.items():
                r, c = own_head
                nr, nc = r + dr, c + dc
                
                # 检查是否是有效移动
                if not (0 <= nr < rows and 0 <= nc < cols) or board_state[nr][nc] in ["own_body", "enemy_body", "enemy_head"]:
                    continue
                
                # 创建新的局面
                new_board = [row[:] for row in board_state]
                new_board[r][c] = "own_body"  # 原来的头变成身体
                new_board[nr][nc] = "own_head"  # 新的头
                
                # 递归计算分数
                score = self.minmax(new_board, (nr, nc), enemy_heads, depth - 1, False)
                max_score = max(max_score, score)
            
            return max_score
        
        else:  # 敌方回合，寻找最小值
            min_score = float('inf')
            
            # 如果没有敌方蛇，跳过
            if not enemy_heads:
                return self.minmax_evaluate(board_state, own_head, enemy_heads, 0)
            
            # 只考虑第一个敌方蛇（简化计算）
            enemy_r, enemy_c = enemy_heads[0]
            
            # 尝试每个可能的移动
            for dr, dc in self.directions.values():
                nr, nc = enemy_r + dr, enemy_c + dc
                
                # 检查是否是有效移动
                if not (0 <= nr < rows and 0 <= nc < cols) or board_state[nr][nc] in ["own_body", "enemy_body", "enemy_head"]:
                    continue
                
                # 创建新的局面
                new_board = [row[:] for row in board_state]
                new_board[enemy_r][enemy_c] = "enemy_body"  # 原来的头变成身体
                new_board[nr][nc] = "enemy_head"  # 新的头
                
                # 更新敌方蛇头位置
                new_enemy_heads = [(nr, nc)] + enemy_heads[1:]
                
                # 递归计算分数
                score = self.minmax(new_board, own_head, new_enemy_heads, depth - 1, True)
                min_score = min(min_score, score)
            
            return min_score
    
    def monte_carlo_simulation(self, board_state, own_head, direction):
        """
        使用蒙特卡洛模拟评估移动的长期收益
        """
        rows = len(board_state)
        cols = len(board_state[0])
        r, c = own_head
        dr, dc = self.directions[direction]
        nr, nc = r + dr, c + dc
        
        # 检查是否是有效移动
        if not (0 <= nr < rows and 0 <= nc < cols) or board_state[nr][nc] in ["own_body", "enemy_body", "enemy_head"]:
            return -float('inf')
        
        # 创建新的局面
        new_board = [row[:] for row in board_state]
        new_board[r][c] = "own_body"  # 原来的头变成身体
        new_board[nr][nc] = "own_head"  # 新的头
        
        # 找到敌方蛇头
        enemy_heads = []
        for r in range(rows):
            for c in range(cols):
                if new_board[r][c] == "enemy_head":
                    enemy_heads.append((r, c))
        
        # 进行多次随机模拟
        total_score = 0
        for _ in range(self.monte_carlo_simulations):
            # 复制当前局面
            sim_board = [row[:] for row in new_board]
            sim_head = (nr, nc)
            sim_enemy_heads = enemy_heads.copy()
            
            # 随机模拟几步
            sim_steps = 5  # 模拟步数
            sim_score = 0
            
            for step in range(sim_steps):
                # 我方随机移动
                valid_moves = []
                for sim_dir, (sim_dr, sim_dc) in self.directions.items():
                    sim_nr, sim_nc = sim_head[0] + sim_dr, sim_head[1] + sim_dc
                    if (0 <= sim_nr < rows and 0 <= sim_nc < cols and 
                        sim_board[sim_nr][sim_nc] not in ["own_body", "enemy_body", "enemy_head"]):
                        valid_moves.append((sim_dir, (sim_nr, sim_nc)))
                
                if not valid_moves:
                    sim_score -= 100  # 无路可走，严重惩罚
                    break
                
                # 随机选择一个有效移动
                sim_dir, (sim_nr, sim_nc) = random.choice(valid_moves)
                
                # 更新局面
                sim_board[sim_head[0]][sim_head[1]] = "own_body"
                sim_board[sim_nr][sim_nc] = "own_head"
                sim_head = (sim_nr, sim_nc)
                
                # 检查是否获得奖励
                if sim_board[sim_nr][sim_nc] in ["score_boost", "speed_boost"]:
                    sim_score += 10
                
                # 敌方随机移动（简化）
                for i, (enemy_r, enemy_c) in enumerate(sim_enemy_heads):
                    enemy_valid_moves = []
                    for enemy_dr, enemy_dc in self.directions.values():
                        enemy_nr, enemy_nc = enemy_r + enemy_dr, enemy_c + enemy_dc
                        if (0 <= enemy_nr < rows and 0 <= enemy_nc < cols and 
                            sim_board[enemy_nr][enemy_nc] not in ["own_body", "enemy_body", "enemy_head"]):
                            enemy_valid_moves.append((enemy_nr, enemy_nc))
                    
                    if enemy_valid_moves:
                        enemy_nr, enemy_nc = random.choice(enemy_valid_moves)
                        sim_board[enemy_r][enemy_c] = "enemy_body"
                        sim_board[enemy_nr][enemy_nc] = "enemy_head"
                        sim_enemy_heads[i] = (enemy_nr, enemy_nc)
                        
                        # 检查是否与敌方相撞
                        if (sim_nr, sim_nc) == (enemy_nr, enemy_nc):
                            sim_score -= 50  # 相撞，严重惩罚
            
            # 评估最终局面
            final_score = sim_score + self.minmax_evaluate(sim_board, sim_head, sim_enemy_heads, 0)
            total_score += final_score
        
        # 返回平均分数
        return total_score / self.monte_carlo_simulations
    
    def determine_best_move(self, board_state):
        """
        综合考虑多种因素，确定最佳移动方向
        """
        # 找到蛇头和敌方蛇头位置
        own_head = None
        enemy_heads = []
        rewards = []
        rows = len(board_state)
        cols = len(board_state[0])
        
        for r in range(rows):
            for c in range(cols):
                if board_state[r][c] == "own_head":
                    own_head = (r, c)
                elif board_state[r][c] == "enemy_head":
                    enemy_heads.append((r, c))
                elif board_state[r][c] in ["score_boost", "speed_boost"]:
                    rewards.append((r, c))
        
        if not own_head:
            return None
        
        # 评估每个可能的移动方向
        move_scores = {}
        
        for direction in self.directions.keys():
            # 1. 安全性评估
            safety = self.evaluate_move_safety(board_state, own_head, direction)
            if safety == -float('inf'):
                move_scores[direction] = -float('inf')  # 不安全的移动
                continue
            
            # 2. MinMax评分
            minmax_score = 0
            if enemy_heads and self.search_depth > 0:
                r, c = own_head
                dr, dc = self.directions[direction]
                nr, nc = r + dr, c + dc
                
                # 检查是否是有效移动
                if 0 <= nr < rows and 0 <= nc < cols and board_state[nr][nc] not in ["own_body", "enemy_body", "enemy_head"]:
                    # 创建新的局面
                    new_board = [row[:] for row in board_state]
                    new_board[r][c] = "own_body"  # 原来的头变成身体
                    new_board[nr][nc] = "own_head"  # 新的头
                    
                    # 使用MinMax算法评估
                    minmax_score = self.minmax(new_board, (nr, nc), enemy_heads, self.search_depth - 1, False)
            
            # 3. 蒙特卡洛模拟
            monte_carlo_score = self.monte_carlo_simulation(board_state, own_head, direction)
            
            # 综合评分
            if minmax_score != -float('inf') and monte_carlo_score != -float('inf'):
                # 权重可以根据实际情况调整
                move_scores[direction] = safety * 0.3 + minmax_score * 0.3 + monte_carlo_score * 0.4
            else:
                move_scores[direction] = -float('inf')
            
            # 避免掉头（除非必要）
            if self.last_direction and direction == self.opposite_directions.get(self.last_direction):
                if move_scores[direction] > -float('inf'):
                    move_scores[direction] -= 20  # 降低掉头的分数
        
        # 选择得分最高的移动
        best_direction = None
        best_score = -float('inf')
        
        for direction, score in move_scores.items():
            if score > best_score:
                best_score = score
                best_direction = direction
        
        # 如果没有可行的移动，尝试随机选择一个非障碍物方向
        if best_direction is None or best_score == -float('inf'):
            possible_directions = []
            for direction, (dr, dc) in self.directions.items():
                r, c = own_head
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and board_state[nr][nc] not in ["own_body", "enemy_body"]:
                    possible_directions.append(direction)
            
            if possible_directions:
                best_direction = random.choice(possible_directions)
                self.log(f"随机选择方向: {best_direction}")
            else:
                self.log("没有可行的移动方向！")
                return None
        
        # 更新最后的方向
        self.last_direction = best_direction
        self.log(f"选择最佳方向: {best_direction}, 得分: {best_score:.2f}")
        
        return best_direction