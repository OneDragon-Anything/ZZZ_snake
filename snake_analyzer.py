import cv2
import numpy as np
import time  # 新增导入time模块
from log import SnakeLogger
from pynput import keyboard  # 导入 pynput 库
from pynput import mouse  # 新增鼠标控制模块
import win32gui  # 新增窗口操作模块
import random  # 导入 random 模块

def hex_to_rgb(hex_color):
    """将十六进制颜色代码转换为 RGB 元组。"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hsv(rgb_color):
    """将 RGB 颜色转换为 HSV 颜色。"""
    normalized_rgb = np.array(rgb_color, dtype=np.float32) / 255.0
    bgr_color = np.uint8([[normalized_rgb[::-1]]])
    hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]
    return hsv_color

class SnakeAnalyzer:
    def __init__(self, logger):  # 修改 __init__ 方法以接收 logger 参数
        self.game_state = "initial"  # 初始状态可以设置为 "initial" 或 "running"
        self.logger = logger  # 保存 logger 实例
        self.grid_colors_hsv = {
            "speed_boost": [72, 149, 235],
            "score_boost": [22, 50, 255],
            "own_head": [4, 213, 255],
            "own_body_tail": [15, 255, 255],
            "enemy_head": [127, 185, 255],
            "enemy_body_tail": [130, 134, 255],
            "grid_light": [97, 85, 221],
            "grid_dark": [94, 247, 203],
        }
        self.snake_direction = None # 初始化蛇头方向为 None
        self.hwnd = None  # 初始化 hwnd 为 None
        self.keyboard_controller = keyboard.Controller() # 初始化 pynput 键盘控制器
        self.mouse_controller = mouse.Controller()  # 新增鼠标控制器
        self.last_click_time = 0  # 新增最后点击时间记录
        self.edge_buffer = 2  # 定义边缘缓冲距离
        self.last_escape_direction = None # 记录上次逃生方向
        self.zigzag_state = 0 # 记录Z字走法的状态
        
        # 导入高级AI决策模块
        self.use_advanced_ai = False
        self.use_minmax_ai = False
        
        try:
            from advanced_snake_ai import AdvancedSnakeAI
            self.advanced_ai = AdvancedSnakeAI(logger)
            self.use_advanced_ai = True
            if logger:
                logger.log("高级AI决策模块已加载")
        except Exception as e:
            if logger:
                logger.log(f"加载高级AI决策模块失败: {str(e)}")
        
        try:
            from minmax_snake_ai import MinMaxSnakeAI
            self.minmax_ai = MinMaxSnakeAI(logger)
            self.use_minmax_ai = True
            if logger:
                logger.log("MinMax AI决策模块已加载")
        except Exception as e:
            if logger:
                logger.log(f"加载MinMax AI决策模块失败: {str(e)}")
                
        if not self.use_advanced_ai and not self.use_minmax_ai:
            if logger:
                logger.log("将使用基础决策算法")

    def analyze_frame(self, frame):
        """
        分析游戏帧，更新游戏状态。
        返回：(game_state, path, special_cells)
        - game_state: 游戏状态
        - path: 计划路径的坐标列表
        - special_cells: 特殊格子的坐标和类型字典
        """
        previous_state = self.game_state
        path = []
        special_cells = {}

        if self.is_game_over(frame):
            self.game_state = "game_over"
            if self.logger and previous_state != "game_over":
                self.logger.log("游戏失败界面检测到！")
        elif self.game_state == "game_over" and not self.is_game_over(frame) and self.detect_snake_head(frame, self.grid_colors_hsv["own_head"][0]):
            self.game_state = "running" # 在游戏失败后，如果检测到蛇头，则认为游戏重新开始
            if self.logger:
                self.logger.log("游戏开始！") # 添加游戏开始的日志输出
        elif self.game_state == "initial" and self.is_game_running(frame):
            self.game_state = "running"

        if self.game_state == "running":
            board_state = self.analyze_board(frame)
            current_time = time.time()  # 获取当前时间

            # 记录特殊格子
            for y in range(len(board_state)):
                for x in range(len(board_state[y])):
                    cell = board_state[y][x]
                    if cell in ['food', 'obstacle', 'own_head', 'own_body', 'enemy_head', 'enemy_body', 'speed_boost', 'score_boost']:
                        special_cells[(x, y)] = cell

            # 获取计划路径
            next_direction = self.determine_next_move(board_state)
            if next_direction:
                # 从蛇头位置开始计算路径
                head_pos = None
                for y in range(len(board_state)):
                    for x in range(len(board_state[y])):
                        if board_state[y][x] == 'own_head':
                            head_pos = (x, y)
                            break
                    if head_pos:
                        break
                
                if head_pos:
                    path = self.calculate_path(board_state, head_pos, next_direction)

                # 控制蛇的移动
                if not hasattr(self, 'last_press_time') or current_time - self.last_press_time > 1:
                    if next_direction:
                        self.control_snake(next_direction)
                        self.last_press_time = current_time
                else:
                    if next_direction:
                        self.control_snake(next_direction)
                        self.last_press_time = current_time

        # 在游戏未开始状态时执行点击
        if self.game_state in ["initial", "game_over"]:
            current_time = time.time()
            if current_time - self.last_click_time > 1:  # 1秒间隔防止连点
                self.click_window_center()
                self.last_click_time = current_time

        return self.game_state, path, special_cells

    def calculate_path(self, board_state, start_pos, direction):
        """
        根据当前方向和游戏状态计算实际路径
        使用与AI决策一致的路径规划逻辑
        """
        # 首先将坐标从(x,y)转换为(row,col)格式，因为board_state是按行列存储的
        start_row, start_col = start_pos[1], start_pos[0]
        head_pos = (start_row, start_col)
        
        # 找到所有奖励的位置
        rewards = []
        rows = len(board_state)
        cols = len(board_state[0])
        
        for r in range(rows):
            for c in range(cols):
                if board_state[r][c] in ["score_boost", "speed_boost"]:
                    rewards.append((r, c))
        
        # 根据direction确定下一步位置
        next_row, next_col = start_row, start_col
        if direction == 'w':  # 上
            next_row -= 1
        elif direction == 's':  # 下
            next_row += 1
        elif direction == 'a':  # 左
            next_col -= 1
        elif direction == 'd':  # 右
            next_col += 1
        
        # 初始化路径，包含起始位置
        path = [start_pos]
        
        # 检查下一步是否会撞墙（地图边缘）
        if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
            self.logger.log(f"警告：下一步将撞墙！方向: {direction}, 位置: ({next_row}, {next_col})")
            # 尝试寻找其他安全方向
            safe_directions = []
            for test_dir in ['w', 's', 'a', 'd']:
                if test_dir == direction:
                    continue  # 跳过当前方向
                
                test_row, test_col = start_row, start_col
                if test_dir == 'w':
                    test_row -= 1
                elif test_dir == 's':
                    test_row += 1
                elif test_dir == 'a':
                    test_col -= 1
                elif test_dir == 'd':
                    test_col += 1
                
                # 检查这个方向是否安全
                if 0 <= test_row < rows and 0 <= test_col < cols and board_state[test_row][test_col] not in ["own_body", "enemy_body", "enemy_head"]:
                    safe_directions.append((test_dir, test_row, test_col))
            
            # 如果有安全方向，选择一个
            if safe_directions:
                new_dir, next_row, next_col = safe_directions[0]  # 选择第一个安全方向
                self.logger.log(f"改变方向避免撞墙: {new_dir}")
                # 更新方向
                direction = new_dir
                # 添加新的下一步位置到路径
                path.append((next_col, next_row))
            else:
                self.logger.log("无法找到安全方向避免撞墙！")
                return path  # 返回只包含起始位置的路径
        elif board_state[next_row][next_col] not in ["own_body", "enemy_body", "enemy_head"]:
            # 如果下一步位置有效，添加到路径（转换回(x,y)格式）
            path.append((next_col, next_row))
        else:
            self.logger.log(f"警告：下一步将撞到障碍物！位置: ({next_row}, {next_col})")
            return path  # 返回只包含起始位置的路径
            
        # 如果有奖励，尝试找到从下一步位置到最近奖励的路径
        if rewards and len(path) > 1:
            closest_reward = None
            min_distance = float('inf')
            for reward_r, reward_c in rewards:
                distance = abs(next_row - reward_r) + abs(next_col - reward_c)
                if distance < min_distance:
                    min_distance = distance
                    closest_reward = (reward_r, reward_c)
            
            if closest_reward:
                # 使用BFS寻找从下一步位置到最近奖励的路径
                bfs_path = self.bfs(board_state, (next_row, next_col), closest_reward)
                if bfs_path and len(bfs_path) > 1:  # 确保路径至少包含两个点
                    # 将BFS路径从(row,col)转换回(x,y)格式，并跳过第一个点（因为已经添加过了）
                    for pos in bfs_path[1:]:
                        path.append((pos[1], pos[0]))
        
        # 如果路径只有起始位置或只有一步，尝试预测几步
        if len(path) <= 2:
            x, y = path[-1]  # 从最后一个位置开始预测
            steps = 5  # 预测未来5步
            
            for _ in range(steps):
                # 根据direction更新位置
                if direction == 'w':  # 上
                    y -= 1
                elif direction == 's':  # 下
                    y += 1
                elif direction == 'a':  # 左
                    x -= 1
                elif direction == 'd':  # 右
                    x += 1
                    
                # 确保坐标在边界内
                if 0 <= x < cols and 0 <= y < rows:
                    # 检查是否会碰到自己的身体或敌人
                    if board_state[y][x] not in ["own_body", "enemy_body", "enemy_head"]:
                        path.append((x, y))
                    else:
                        break
                else:
                    # 预测到会撞墙，停止预测
                    self.logger.log(f"预测路径将撞墙！位置: ({x}, {y})")
                    break
        
        return path

    def click_window_center(self):
        """点击游戏窗口中心区域"""
        try:
            hwnd = win32gui.FindWindow(None, '绝区零')
            if hwnd:
                # 获取窗口客户区坐标
                left, top, right, bottom = win32gui.GetClientRect(hwnd)
                # 转换为屏幕坐标
                left, top = win32gui.ClientToScreen(hwnd, (left, top))
                right, bottom = win32gui.ClientToScreen(hwnd, (right, bottom))
                # 计算中心点
                center_x = (left + right) // 2
                center_y = (top + bottom) // 2

                # 执行点击操作（模仿键盘控制的延迟节奏）
                self.mouse_controller.position = (center_x, center_y)
                time.sleep(0.05)
                self.mouse_controller.press(mouse.Button.left)
                time.sleep(0.05)
                self.mouse_controller.release(mouse.Button.left)
                self.logger.log(f"窗口中心点击 @ ({center_x}, {center_y})")
        except Exception as e:
            self.logger.log(f"点击失败: {str(e)}")

    def is_game_over(self, frame):
        """
        检测当前帧是否是游戏失败界面.
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_failure = np.array([100, 171, 69])  # 基于之前的 HSB 转换
        upper_failure = np.array([118, 211, 109])
        mask = cv2.inRange(hsv_frame, lower_failure, upper_failure)
        failure_pixels = cv2.countNonZero(mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        return failure_pixels > total_pixels * 0.1  # 阈值可以根据实际情况调整

    def is_game_running(self, frame):
        """
        检测当前帧是否是游戏运行界面.
        """
        # 这里可以添加检测游戏运行界面的逻辑，例如检测网格线、蛇或食物
        return True # 暂时返回 True，你需要根据实际情况实现

    def is_start_screen(self, frame):
        """
        检测当前帧是否是开始界面.
        """
        # 这里可以添加检测开始界面的逻辑，例如检测 "START" 文本
        return False # 暂时返回 False，你需要根据实际情况实现

    def detect_snake_head(self, frame, target_hue):
        """
        检测屏幕中是否存在指定色调的蛇头图案.
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 定义目标色调的范围 (允许一定的容差)
        lower_bound = np.array([target_hue - 5, 50, 50])  # 调整饱和度和明度的下限
        upper_bound = np.array([target_hue + 5, 255, 255]) # 调整饱和度和明度的上限
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        head_pixels = cv2.countNonZero(mask)
        # 需要根据实际蛇头的大小调整阈值
        threshold = 30  # 这是一个初始猜测值，可能需要调整
        return head_pixels > threshold

    def analyze_board(self, frame):
        """
        分析游戏棋盘，识别每个格子的内容并进行日志记录。
        """
        grid_height = 25
        grid_width = 29
        frame_height, frame_width, _ = frame.shape
        cell_size_height = frame_height / grid_height
        cell_size_width = frame_width / grid_width
        # 假设格子是正方形，取较小的尺寸作为 cell_size
        cell_size = min(cell_size_height, cell_size_width)

        board = [[None for _ in range(grid_width)] for _ in range(grid_height)]

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        own_head_hue = self.grid_colors_hsv["own_head"][0]
        own_tail_hue = self.grid_colors_hsv["own_body_tail"][0]
        enemy_head_hue = self.grid_colors_hsv["enemy_head"][0]
        enemy_tail_hue = self.grid_colors_hsv["enemy_body_tail"][0]
        grid_light_hue = self.grid_colors_hsv["grid_light"][0]
        grid_dark_hue = self.grid_colors_hsv["grid_dark"][0]
        speed_boost_hue = self.grid_colors_hsv["speed_boost"][0]
        score_boost_hue = self.grid_colors_hsv["score_boost"][0]

        # 在 analyze_board 方法中新增颜色检测逻辑
        for r in range(grid_height):
            for c in range(grid_width):
                # 计算当前格子的像素坐标
                x = int(c * cell_size)
                y = int(r * cell_size)
                # 检查是否在边缘区域    
                # 为了更准确地获取格子的颜色，我们可以取格子中心点的颜色
                center_x = int(c * cell_size + cell_size / 2)
                center_y = int(r * cell_size + cell_size / 2)

                # 获取中心像素的 HSV 值
                # 确保中心坐标在图像边界内
                if 0 <= center_y < frame_height and 0 <= center_x < frame_width:
                    hue = hsv_frame[center_y, center_x, 0]
                else:
                    hue = -1 # 或者其他表示越界的值

                found = False
                # 检查蛇头 (处理边框)
                head_detected = False
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        check_center_x = center_x + j
                        check_center_y = center_y + i
                        if 0 <= check_center_y < frame_height and 0 <= check_center_x < frame_width:
                            hue_check = hsv_frame[check_center_y, check_center_x, 0]
                            if hue_check == own_head_hue:
                                board[r][c] = "own_head"
                                #self.logger.log(f"我方蛇头坐标: ({r}, {c})")
                                head_detected = True
                                found = True
                                break
                    if head_detected:
                        break
                if found:
                    continue

                if own_head_hue <= hue <= own_tail_hue:
                    board[r][c] = "own_body"
                    found = True
                elif hue == enemy_head_hue:
                    board[r][c] = "enemy_head"
                    #self.logger.log(f"敌蛇头坐标: ({r}, {c})")
                    found = True
                elif enemy_head_hue <= hue <= enemy_tail_hue:
                    board[r][c] = "enemy_body"
                    found = True
                elif abs(int(hue) - int(speed_boost_hue)) <= 10:  # 加速道具，允许±10误差
                    board[r][c] = "speed_boost"
                    #self.logger.log(f"加速坐标: ({r}, {c})")
                    found = True
                elif abs(int(hue) - int(score_boost_hue)) <= 10:  # 分数道具，允许±10误差
                    board[r][c] = "score_boost"
                    #self.logger.log(f"分数坐标: ({r}, {c})")
                    found = True
                elif (hue <= grid_light_hue + 10 or hue >= grid_dark_hue - 10):  # 空白格子，允许±10误差
                    board[r][c] = "blank"
                    found = True

                if not found:
                    board[r][c] = "unknown_reward"
                    # 不再输出未知奖励的 log

        return board

    def control_snake(self, direction):
        """
        模拟按下指定的按键来控制蛇的移动 (使用 pynput)。
        direction: 'w', 's', 'a', 或 'd'，代表上、下、左、右。
        """
        if direction in ['w', 's', 'a', 'd']:
            self.logger.log(f"按下了按键 (pynput): ({direction})")
            self.keyboard_controller.press(keyboard.KeyCode.from_char(direction))
            time.sleep(0.05)  # 可以根据需要调整延迟
            self.keyboard_controller.release(keyboard.KeyCode.from_char(direction))
            time.sleep(0.05)
        else:
            self.logger.log(f"无效的控制方向: {direction}")

    def bfs(self, board_state, start, end):
        queue = [(start, [start])]
        visited = {start}
        rows = len(board_state)
        cols = len(board_state[0])

        while queue:
            (r, c), path = queue.pop(0)

            if (r, c) == end:
                return path

            # 定义可能的移动方向：上、下、左、右
            moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]

            for dr, dc in moves:
                nr, nc = r + dr, c + dc

                # 检查是否在棋盘边界内
                if 0 <= nr < rows and 0 <= nc < cols and board_state[nr][nc] not in ["own_body"]:
                    neighbor = (nr, nc)
                    if neighbor not in visited:
                        # 检查是否是边缘格子（更严格的边界检查）
                        is_edge_neighbor = (nr == 0 or nr == rows - 1 or nc == 0 or nc == cols - 1)
                        
                        # 检查是否靠近边缘（使用edge_buffer）
                        is_near_edge = (nr < self.edge_buffer or nr >= rows - self.edge_buffer or
                                       nc < self.edge_buffer or nc >= cols - self.edge_buffer)

                        is_near_enemy = False
                        # 检查周围一格是否有敌人蛇
                        for i in range(-1, 2):
                            for j in range(-1, 2):
                                if abs(i) + abs(j) <= 1: # 检查上下左右和当前格子
                                    er, ec = nr + i, nc + j
                                    if 0 <= er < rows and 0 <= ec < cols and board_state[er][ec] in ["enemy_head", "enemy_body"]:
                                        is_near_enemy = True
                                        break
                            if is_near_enemy:
                                break

                        # 优先选择不靠近边缘和敌人的路径
                        if not is_near_edge and not is_near_enemy:
                            visited.add(neighbor)
                            new_path = list(path)
                            new_path.append(neighbor)
                            queue.append((neighbor, new_path))
                        elif not is_edge_neighbor and (is_near_edge or is_near_enemy): 
                            # 允许靠近边缘或敌人，但不允许直接到达边缘，且优先级较低
                            visited.add(neighbor)
                            new_path = list(path)
                            new_path.append(neighbor)
                            queue.append((neighbor, new_path))

        # 如果没有找到理想路径，尝试放宽条件再次搜索
        if end not in visited:
            queue = [(start, [start])]
            visited = {start}
            
            while queue:
                (r, c), path = queue.pop(0)
                
                if (r, c) == end:
                    return path
                    
                moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
                
                for dr, dc in moves:
                    nr, nc = r + dr, c + dc
                    
                    # 只检查基本边界和障碍物
                    if 0 <= nr < rows and 0 <= nc < cols and board_state[nr][nc] not in ["own_body"]:
                        neighbor = (nr, nc)
                        if neighbor not in visited:
                            visited.add(neighbor)
                            new_path = list(path)
                            new_path.append(neighbor)
                            queue.append((neighbor, new_path))
        
        return None # 如果没有找到路径，返回 None

    # 新增逃生路径检测方法
    def find_escape_path(self, board_state, head_pos):
        """寻找最近的逃生路径，优先考虑远离边界的方向"""
        rows = len(board_state)
        cols = len(board_state[0])
        r, c = head_pos
        escape_moves = []
        center_r, center_c = rows // 2, cols // 2

        # 检查上、下、左、右四个方向
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        safe_moves = []
        risky_moves = []

        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            # 首先检查是否在边界内且不会撞到障碍物
            if 0 <= nr < rows and 0 <= nc < cols and board_state[nr][nc] not in ["own_body", "enemy_body"]:
                # 计算该位置到中心的距离
                dist_to_center = abs(nr - center_r) + abs(nc - center_c)
                # 检查是否是边缘或靠近边缘
                is_at_edge = (nr == 0 or nr == rows - 1 or nc == 0 or nc == cols - 1)
                is_near_edge = (nr < self.edge_buffer or nr >= rows - self.edge_buffer or
                               nc < self.edge_buffer or nc >= cols - self.edge_buffer)
                
                # 根据位置分类移动
                if is_at_edge:
                    # 最危险的移动，只有在没有其他选择时才考虑
                    risky_moves.append((dr, dc, dist_to_center))
                elif is_near_edge:
                    # 次危险的移动
                    risky_moves.append((dr, dc, dist_to_center - 10))  # 给予一定优先级
                else:
                    # 安全的移动
                    safe_moves.append((dr, dc, dist_to_center))
        
        # 优先选择安全的移动，按照到中心的距离排序（距离越小越好）
        if safe_moves:
            safe_moves.sort(key=lambda x: x[2])
            for dr, dc, _ in safe_moves:
                escape_moves.append((dr, dc))
        
        # 如果没有安全移动，考虑风险移动
        if not escape_moves and risky_moves:
            risky_moves.sort(key=lambda x: x[2])
            for dr, dc, _ in risky_moves:
                escape_moves.append((dr, dc))
                
        if escape_moves:
            self.logger.log(f"找到逃生路径，共{len(escape_moves)}个方向可选")
        else:
            self.logger.log("警告：找不到任何逃生路径！")

        return escape_moves

    # 决策算法，集成了高级AI、MinMax AI和基础策略
    def determine_next_move(self, board_state):
        # 检查是否有敌方蛇头
        has_enemy = False
        for r in range(len(board_state)):
            for c in range(len(board_state[r])):
                if board_state[r][c] == "enemy_head":
                    has_enemy = True
                    break
            if has_enemy:
                break
        
        # 根据游戏状态选择最合适的AI
        # 如果有敌方蛇，优先使用MinMax AI（更适合对抗场景）
        if has_enemy and self.use_minmax_ai:
            try:
                self.logger.log("使用MinMax AI进行决策（对抗模式）")
                direction = self.minmax_ai.determine_best_move(board_state)
                if direction:
                    self.snake_direction = direction
                    return direction
            except Exception as e:
                if self.logger:
                    self.logger.log(f"MinMax AI决策失败: {str(e)}")
        
        # 如果没有敌方蛇或MinMax AI失败，尝试使用高级AI
        if self.use_advanced_ai:
            try:
                self.logger.log("使用高级AI进行决策")
                direction = self.advanced_ai.determine_best_move(board_state)
                if direction:
                    self.snake_direction = direction
                    return direction
            except Exception as e:
                if self.logger:
                    self.logger.log(f"高级AI决策失败: {str(e)}")
        
        # 如果高级AI都失败，回退到基础策略
        self.logger.log("使用基础策略进行决策")
        
        # 基础决策策略（原有代码）
        head_row, head_col = None, None
        rewards = []
        rows = len(board_state)
        cols = len(board_state[0])

        # 找到蛇头和所有奖励的位置
        for r in range(rows):
            for c in range(cols):
                if board_state[r][c] == "own_head":
                    head_row, head_col = r, c
                elif board_state[r][c] in ["score_boost"]: # 只以分数奖励为目标
                    rewards.append((r, c))

        if head_row is None:
            return None

        # 检查是否在边缘或即将撞墙
        is_at_edge = (head_row == 0 or head_row == rows - 1 or head_col == 0 or head_col == cols - 1)
        is_near_edge = (head_row < self.edge_buffer or head_row >= rows - self.edge_buffer or
                         head_col < self.edge_buffer or head_col >= cols - self.edge_buffer)

        # 如果已经在边缘，立即采取紧急措施
        if is_at_edge:
            self.logger.log("警告：蛇头已在地图边缘！")
            possible_directions = {'w': (-1, 0), 's': (1, 0), 'a': (0, -1), 'd': (0, 1)}
            safe_directions = []
            
            # 寻找所有安全方向（不会撞墙的方向）
            for direction, (dr, dc) in possible_directions.items():
                next_r, next_c = head_row + dr, head_col + dc
                if 0 <= next_r < rows and 0 <= next_c < cols and board_state[next_r][next_c] not in ["own_body", "enemy_body"]:
                    # 计算这个方向离中心的距离
                    center_distance = abs(next_r - rows//2) + abs(next_c - cols//2)
                    safe_directions.append((direction, center_distance))
            
            # 按照离中心距离排序，优先选择向中心移动的方向
            if safe_directions:
                safe_directions.sort(key=lambda x: x[1])
                chosen_direction = safe_directions[0][0]
                self.logger.log(f"紧急避险：选择方向 {chosen_direction} 远离边缘")
                self.snake_direction = chosen_direction
                return chosen_direction
        
        # 如果靠近边缘但还未到达，尝试远离
        elif is_near_edge:
            possible_directions = {'w': (-1, 0), 's': (1, 0), 'a': (0, -1), 'd': (0, 1)}
            away_directions = []
            
            for direction, (dr, dc) in possible_directions.items():
                next_r, next_c = head_row + dr, head_col + dc
                # 首先确保不会撞墙
                if 0 <= next_r < rows and 0 <= next_c < cols and board_state[next_r][next_c] not in ["own_body", "enemy_body"]:
                    # 检查这个方向是否会让蛇远离边缘
                    if head_row < self.edge_buffer and next_r > head_row:
                        away_directions.append(direction)
                    elif head_row >= rows - self.edge_buffer and next_r < head_row:
                        away_directions.append(direction)
                    elif head_col < self.edge_buffer and next_c > head_col:
                        away_directions.append(direction)
                    elif head_col >= cols - self.edge_buffer and next_c < head_col:
                        away_directions.append(direction)
                    # 如果不是靠近某一边缘，而是在角落，添加任何指向中心的方向
                    elif abs(next_r - rows//2) + abs(next_c - cols//2) < abs(head_row - rows//2) + abs(head_col - cols//2):
                        away_directions.append(direction)

            if away_directions:
                chosen_direction = random.choice(away_directions)
                self.logger.log(f"远离边缘：选择方向 {chosen_direction}")
                self.snake_direction = chosen_direction
                return chosen_direction

        # 寻找最近的奖励
        if rewards:
            closest_reward = None
            min_distance = float('inf')
            for reward_r, reward_c in rewards:
                distance = abs(head_row - reward_r) + abs(head_col - reward_c)
                if distance < min_distance:
                    min_distance = distance
                    closest_reward = (reward_r, reward_c)

            if closest_reward:
                shortest_path = self.bfs(board_state, (head_row, head_col), closest_reward)
                if shortest_path and len(shortest_path) > 1:
                    next_cell_row, next_cell_col = shortest_path[1]
                    required_direction = None
                    if next_cell_row < head_row:
                        required_direction = 'w'
                    elif next_cell_row > head_row:
                        required_direction = 's'
                    elif next_cell_col < head_col:
                        required_direction = 'a'
                    elif next_cell_col > head_col:
                        required_direction = 'd'

                    if required_direction and required_direction != self.snake_direction:
                        self.snake_direction = required_direction
                        return required_direction
                    else:
                        return None # 方向一致，不需要操作

        # Z字探索策略 (当没有奖励时且不在边缘附近)
        if not rewards and not is_near_edge:
            if self.zigzag_state == 0:
                direction = 'd'
                if head_col == cols - 1 - self.edge_buffer:
                    self.zigzag_state = 1
            elif self.zigzag_state == 1:
                direction = 's'
                if head_row == rows - 1 - self.edge_buffer:
                    self.zigzag_state = 2
            elif self.zigzag_state == 2:
                direction = 'a'
                if head_col == self.edge_buffer:
                    self.zigzag_state = 3
            elif self.zigzag_state == 3:
                direction = 's'
                if head_row < rows - 1 - self.edge_buffer + 5: # 稍微多走一点
                    direction = 's'
                else:
                    self.zigzag_state = 0 # 重置

            if 0 <= head_row + {'w': -1, 's': 1, 'a': 0, 'd': 0}.get(direction, 0) < rows and \
               0 <= head_col + {'w': 0, 's': 0, 'a': -1, 'd': 1}.get(direction, 0) < cols and \
               board_state[head_row + {'w': -1, 's': 1, 'a': 0, 'd': 0}.get(direction, 0)][head_col + {'w': 0, 's': 0, 'a': -1, 'd': 1}.get(direction, 0)] not in ["own_body", "enemy_body"]:
                if direction and direction != self.snake_direction:
                    self.snake_direction = direction
                    return direction
                elif direction:
                    return direction


        # 如果没有奖励或者无法到达奖励，尝试寻找安全的逃生方向
        head_pos = (head_row, head_col)
        escape_directions = self.find_escape_path(board_state, head_pos)

        if escape_directions:
            # 避免立即掉头
            if self.last_escape_direction and len(escape_directions) > 1:
                opposite_direction = {'w': (1, 0), 's': (-1, 0), 'a': (0, 1), 'd': (0, -1)}
                valid_escape_directions = []
                for dr, dc in escape_directions:
                    if (dr, dc) != opposite_direction.get(self.last_escape_direction):
                        valid_escape_directions.append((dr, dc))
                if valid_escape_directions:
                    escape_directions = valid_escape_directions

            if escape_directions:
                dr, dc = random.choice(escape_directions)
                if dr == -1:
                    chosen_direction = 'w'
                elif dr == 1:
                    chosen_direction = 's'
                elif dc == -1:
                    chosen_direction = 'a'
                elif dc == 1:
                    chosen_direction = 'd'

                if chosen_direction:
                    self.last_escape_direction = chosen_direction
                    if chosen_direction != self.snake_direction:
                        self.snake_direction = chosen_direction
                        return chosen_direction
                    else:
                        return chosen_direction # 即使方向相同也返回，避免卡住
        else:
            self.logger.log("找不到任何安全的移动方向！")
            return None

if __name__ == '__main__':
    from log import SnakeLogger  # 确保这里导入 SnakeLogger
    import time

    logger = SnakeLogger()
    analyzer = SnakeAnalyzer(logger)

    try:
        frame = cv2.imread("image_f1f241.jpg") # 尝试加载你上传的截图
        if frame is None:
            print("无法加载图像 'image_f1f241.jpg'，请确保文件存在于当前目录下。")
        else:
            game_state = analyzer.analyze_frame(frame)
            print(f"当前游戏状态: {game_state}")

            if game_state == "running":
                board_state = analyzer.analyze_board(frame)
                # 打印棋盘状态 (仅为演示)
                for row in board_state:
                    print(row)

                # 示例如何控制蛇 (你需要根据你的逻辑来决定何时以及按哪个键)
                # 在这里可以取消注释并根据需要调用 control_snake
                # 例如：
                # if analyzer.snake_direction:
                #     analyzer.control_snake(analyzer.snake_direction)

    except Exception as e:
        print(f"发生错误: {e}")