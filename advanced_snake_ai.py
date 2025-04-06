import numpy as np
import heapq
import random

class AdvancedSnakeAI:
    """
    高级贪吃蛇AI决策算法实现，包含A*寻路、Voronoi区域控制和敌方路径预测
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
        queue = [(nr, nc)]
        visited.add((nr, nc))
        
        while queue:
            curr_r, curr_c = queue.pop(0)
            
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
        
        # 计算Voronoi控制区域
        control_map = None
        if enemy_heads:
            control_map = self.calculate_voronoi(board_state, own_head, enemy_heads)
        
        # 预测敌方移动
        predicted_enemy_positions = []
        if enemy_heads:
            predicted_enemy_positions = self.predict_enemy_moves(board_state, enemy_heads)
        
        # 评估每个可能的移动方向
        move_scores = {}
        for direction, (dr, dc) in self.directions.items():
            r, c = own_head
            nr, nc = r + dr, c + dc
            
            # 检查是否在边界内且不是障碍物
            if not (0 <= nr < rows and 0 <= nc < cols) or board_state[nr][nc] in ["own_body", "enemy_body"]:
                move_scores[direction] = -float('inf')  # 不可行的移动
                continue
            
            # 初始分数
            score = 0
            
            # 1. 安全性评估
            safety = self.evaluate_move_safety(board_state, own_head, direction)
            if safety == -float('inf'):
                move_scores[direction] = -float('inf')  # 不安全的移动
                continue
            
            score += safety * 0.5  # 安全性权重
            
            # 2. 奖励吸引力
            if rewards:
                # 找到最近的奖励
                min_distance = float('inf')
                closest_reward = None
                
                for reward_r, reward_c in rewards:
                    distance = abs(nr - reward_r) + abs(nc - reward_c)
                    if distance < min_distance:
                        min_distance = distance
                        closest_reward = (reward_r, reward_c)
                
                if closest_reward:
                    # 使用A*寻找路径
                    path = self.a_star_search(board_state, (nr, nc), closest_reward)
                    if path:
                        # 路径存在，增加分数
                        score += self.food_weight * (50 / (len(path) + 1))  # 距离越近分数越高
            
            # 3. 控制区域评估
            if control_map is not None:
                # 移动后的位置在控制图中的值
                control_value = control_map[nr][nc]
                score += control_value * self.space_weight
            
            # 4. 避开预测的敌方位置
            for pred_r, pred_c in predicted_enemy_positions:
                if (nr, nc) == (pred_r, pred_c):
                    score -= self.danger_weight * 5  # 大幅降低分数
                elif abs(nr - pred_r) + abs(nc - pred_c) <= 1:
                    score -= self.danger_weight * 2  # 相邻位置也降低分数
            
            # 5. 避免掉头（除非必要）
            if self.last_direction and direction == self.opposite_directions.get(self.last_direction):
                score -= 20  # 降低掉头的分数
            
            # 记录最终分数
            move_scores[direction] = score
        
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