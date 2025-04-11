import cv2
import numpy as np
import time
from log.debug_helper import DebugHelper  # 新增导入
from model.image_cell import ImageCell    # <= 这里改成这样

class BoardAnalyzer:
    """负责分析棋盘状态的类"""
    def __init__(self, logger=None):
        self.logger = logger
        self.board = None
        self.special_cells = {}
        self.last_frame = None  # 上一帧画面
        self.last_target_pos = []  # 存储多个终点关注单位的坐标
        self.head_position = None  # 蛇头精准坐标 (x,y)
        self.head_direction = None  # 蛇头运动方向 ('up','down','left','right')
        self.is_gameover = False
        self.is_running = False
        self.last_gameover_check = 0  # 上次检测gameover的时间戳
        self.last_no_head_save_time = 0
        self.last_eye_position = None  # 上一帧蛇眼重心坐标
        self.eye_position = None       # 当前帧蛇眼重心坐标
        self.last_eye_time = 0        # 最后检测到眼睛的时间戳
        self.last_dir_change_pos = None  # 最后方向变化时的位置差(dx,dy)
        self.last_dir_change_time = 0    # 最后方向变化的时间戳
        self.eye_velocity = (0, 0)     # 蛇眼移动速度 (vx,vy) 像素/秒

        self.grid_colors_hsv = {
            "speed_boost": [72, 149, 235],
            "score_boost": [20, 156, 255],
            "score_boost2": [20, 156, 255],
            "own_head": [4, 213, 255],
            "own_body_tail": [15, 255, 255],
            "enemy_head": [127, 185, 255],
            "enemy_body_tail": [130, 134, 255],
            "enemy2_head": [150, 185, 255],
            "enemy2_body_tail": [134, 134, 255],
            "enemy3_head": [0, 185, 255],
            "enemy3_body_tail": [156, 134, 255],
            "grid_light": [88, 85, 221],
            "grid_dark": [99, 247, 203],
            "game_over":  [109, 192, 88],
        }
        
        # 预设颜色参数
        self.own_head = self.grid_colors_hsv["own_head"][0]
        self.own_tail = self.grid_colors_hsv["own_body_tail"][0]
        self.enemy_head = self.grid_colors_hsv["enemy_head"][0]
        self.enemy_tail = self.grid_colors_hsv["enemy_body_tail"][0]
        self.enemy2_head = self.grid_colors_hsv["enemy2_head"][0]
        self.enemy2_tail = self.grid_colors_hsv["enemy2_body_tail"][0]
        self.enemy3_head = self.grid_colors_hsv["enemy3_head"][0]
        self.enemy3_tail = self.grid_colors_hsv["enemy3_body_tail"][0]
        self.speed_boost = self.grid_colors_hsv["speed_boost"][0]
        self.score_boost = self.grid_colors_hsv["score_boost"][0]
        self.score_boost2 = self.grid_colors_hsv["score_boost2"][0]
        self.grid_light = self.grid_colors_hsv["grid_light"][0]
        self.grid_dark = self.grid_colors_hsv["grid_dark"][0]
        self.game_over = self.grid_colors_hsv["game_over"][0]


    def analyze_board(self, board, board_image=None, image_format='HSV', last_key_direction=None):
        """分析棋盘图像并初始化棋盘状态
        :param board_image: 输入的棋盘图像(HSV格式)
        :return: Board对象
        """
        
        self.head_direction = last_key_direction
        # 创建Board对象并设置图像
        if board_image is not None:
            board.set_hsv_image(board_image, image_format)
        
        # 存储棋盘对象
        self.board = board
        
        # 检测gameover状态（每10秒检测一次）
        current_time = time.time()
        if current_time - self.last_gameover_check >= 10:
            gameover_result = self.analyze_color_move_direction(
                [self.game_over, 0, 0], 
                [0, 255, 255], 
            )
            # 检查坐标是否在474,426正负5像素范围内
            if gameover_result:
                _, (cx, cy),_ = gameover_result
                if 469 <= cx <= 479 and 421 <= cy <= 431:
                    self.is_gameover = True
                    return board
            
            self.last_gameover_check = current_time
            self.is_gameover = False
            
        # 检测棋盘四个角的颜色是否在有效范围内
        h, w = board.hsv_image.shape[:2]
        corners = [
            board.hsv_image[0, 0, 0],    # 左上角
            board.hsv_image[0, w-1, 0],  # 右上角
            board.hsv_image[h-1, 0, 0],  # 左下角
            board.hsv_image[h-1, w-1, 0] # 右下角
        ]
        
        # 检查四个角的H值是否都在grid_light-10到grid_dark+10范围内
        valid_corners = all(
            (self.grid_light - 10) <= h_val <= (self.grid_dark + 10)
            for h_val in corners
        )
        
        # 如果四个角颜色不满足条件，初始化状态
        if not valid_corners:
            self.is_running = False
            self.head_position = None
            self.head_direction = None
            self.last_frame = None
            return board
            
        # 进行深度分析
        self.deep_analysis()

        # 进行前后帧分析
        self.analyze_frame_changes(board_image)
        
        # 检测蛇头状态
        if "own_head" not in self.special_cells:
            # 初始化状态
            print("没有找到蛇头")
            now = time.time()
            if now - self.last_no_head_save_time > 1:
                try:
                    file_path = DebugHelper.save_image(self.board.bgr_image, prefix="nosnakehead")
                    self.last_no_head_save_time = now
                    if self.logger:
                        self.logger.log(f"[DEBUG] 未检测到蛇头，截图已保存至: {file_path}")
                except Exception as e:
                    if self.logger:
                        self.logger.log(f"[DEBUG] 未检测到蛇头，保存截图异常: {e}")
            self.is_running = False
            self.head_position = None
            self.head_direction = None
            self.last_frame = None
        else:
            # 设置运行状态
            self.is_running = True


        self.special_cells = {
            k: v for k, v in self.special_cells.items() if v
        }

        board.special_cells = self.special_cells
        board.head_position  = self.head_position
        board.direction  = self.head_direction
        return board

    def deep_analysis(self):
        if not self.board:
            return
        
        # 初始化，并只继承上次的蛇身，不继承蛇头
        self.special_cells = {}
        previous = self.board.special_cells if hasattr(self.board, "special_cells") else {}
        inherited_body_cells = []
        for cell in previous.get("own_body", []):
            cell.cell_type = "own_body"
            self.add_to_special_cells("own_body", cell)
            inherited_body_cells.append(cell)

        # 全盘扫描
        for row in self.board.cells:
            for cell in row:
                # 如果是继承的蛇身格子，跳过识别
                if cell in inherited_body_cells:
                    continue
                cell_type = self.determine_cell_type(cell.center_color)
                cell.cell_type = cell_type
                self.add_to_special_cells(cell_type, cell)

        # 深度分析未知格
        unknow_cells = self.special_cells.get("unknow", [])
        if unknow_cells:
            for cell in unknow_cells:
                self.deep_analysis_cell(cell)
        
    def deep_analysis_cell(self, cell):
        """
        深度分析指定的单个格子
        功能：对单个格子进行更精确的颜色分析，根据颜色分布确定格子类型
        参数：
            cell: 要分析的格子对象
        返回：
            更新后的格子类型字符串
        """
        # 检查输入参数有效性
        if not self.board or not cell:
            return None
            
        # 获取格子内3x3区域的颜色分布字典
        # key: 颜色H值, value: 该颜色出现的次数
        color_dict = cell.get_color_dict(self.board.hsv_image, 3)
        
        if color_dict:
            # 按颜色出现频率从高到低排序
            final_type = "empty"  # 默认最终类型为空
            final_h_value = None
            for h_value, count in sorted(color_dict.items(), key=lambda x: x[1], reverse=True):
                # 根据颜色H值确定格子类型
                new_type = self.determine_cell_type((h_value, 0, 0))
                if new_type == "empty":
                    final_h_value = h_value
                    continue
                # 关键逻辑：如果找到非empty/unknow的类型，立即采用并返回
                if new_type not in ["empty", "unknow"]:
                    # 从原类型列表中移除该格子
                    self.remove_from_special_cells(cell.cell_type, cell)
                    # 更新格子的中心颜色和类型
                    cell.center_color = (h_value, 0, 0)
                    cell.cell_type = new_type
                    # 添加到新类型列表中
                    self.add_to_special_cells(new_type, cell)
                    return new_type  # 找到确定类型后立即返回
            
            # 如果遍历完所有颜色都只有empty/unknow，则强制设置为empty
            self.remove_from_special_cells(cell.cell_type, cell)
            cell.center_color = (final_h_value, 0, 0)  # 使用最后一个h_value
            cell.cell_type = final_type
            return final_type
            
        # 如果没有颜色数据，保持原类型不变
        return cell.cell_type
        
    def determine_own_tail(self):
        """
        找到蛇尾的格子
        """
        if "own_body" not in self.special_cells:
            return None

        result = self.analyze_color_move_direction([self.own_tail,0,255], [0,255,50], preset_direction="all")
        if not result:
            return None

        _, _, edge_points = result

        candidate_cells = []
        own_tail_scores = []

        color_tolerance = 1  # 容差值

        for key in ['left', 'right', 'up', 'down']:
            point = edge_points.get(key)
            if not point:
                continue
            cell = self.board.get_cell_by_position(point[0], point[1])
            if not cell:
                continue
            color_dict = cell.get_color_dict(self.board.hsv_image, 3)
            score = 0
            if color_dict:
                for h_value, cnt in color_dict.items():
                    # 如果颜色在阈值范围内，计入得分
                    if abs(int(h_value) - int(self.own_tail)) <= color_tolerance:
                        score += cnt
            if score > 0:
                candidate_cells.append(cell)
                own_tail_scores.append(score)

        if not candidate_cells:
            return None

        min_index = np.argmin(own_tail_scores)
        tail_cell = candidate_cells[min_index]

        # 更新special_cells
        self.remove_from_special_cells(tail_cell.cell_type, tail_cell)
        tail_cell.cell_type = "own_tail"
        self.add_to_special_cells("own_tail", tail_cell)

        return tail_cell


    def determine_cell_type(self, hsv_color):
        """
        根据HSV颜色确定格子类型
        :param hsv_color: HSV颜色值
        :return: 格子类型
        """
        h, s, v = hsv_color
        
        # 判断各种类型
        # if  h == self.own_head:
        #     return "own_head"
        if self.own_head <= h <= self.own_tail:
            return "own_body"
        elif h in [self.enemy_head,self.enemy2_head,self.enemy3_head]:
            return "enemy_head"
        elif (self.enemy2_tail <= h <= self.enemy2_head) or (self.enemy_tail >= h >= self.enemy_head) or (self.enemy2_tail <= h):
            return "enemy_body"
        elif h == self.speed_boost:
            return "speed_boost"
        elif (self.score_boost-5) <= h <= (self.score_boost+5):
            return "score_boost"
        elif (self.score_boost2-5) <= h <= (self.score_boost2+5):
            return "score_boost"
        elif (self.grid_light) <=  h <= (self.grid_dark):
            return "empty"
        else:
            return "unknow"
            
    def add_to_special_cells(self, cell_type, cell):
        """
        将单元格添加到special_cells字典中，避免坐标重复
        """
        if cell is None:
            return  # 防止None被添加进来

        # 可选！严格类型判断
        if not isinstance(cell, ImageCell):
            return  # 不是合法棋盘格，跳过

        if cell_type not in self.special_cells:
            self.special_cells[cell_type] = []

        # 先移除坐标相同的已有cell
        self.special_cells[cell_type] = [
            c for c in self.special_cells[cell_type]
            if not (c.row == cell.row and c.col == cell.col)
        ]

        # 添加新cell（坐标唯一）
        self.special_cells[cell_type].append(cell)

    def remove_from_special_cells(self, cell_type, cell):
        """
        从special_cells字典中移除单元格
        :param cell_type: 单元格类型
        :param cell: 要移除的单元格对象
        """
        if cell_type in self.special_cells and cell in self.special_cells[cell_type]:
            self.special_cells[cell_type].remove(cell)
        
    def analyze_frame_changes(self, current_frame):
        """
        分析前后帧的变化，并根据蛇眼判断蛇头格子

        :param current_frame: 当前帧HSV图像
        :return: True
        """
        current_time = time.time()
        eyes = self.find_snake_eye()
        if eyes:
            avg_x = int(np.mean([p[0] for p in eyes]))
            avg_y = int(np.mean([p[1] for p in eyes]))
            # 识别成功，更新历史缓存
            self.last_eye_position = self.eye_position
            self.eye_position = (avg_x, avg_y)
            self.last_eye_time = current_time
        else:
            # 未检测到眼睛时，使用速度预测
            if hasattr(self, 'last_dir_change_pos') and hasattr(self, 'last_eye_time'):
                time_diff = current_time - self.last_eye_time
                if time_diff < 0.5:  # 仅在短时间内预测
                    pred_x = self.eye_position[0] + self.last_dir_change_pos[0] * time_diff
                    pred_y = self.eye_position[1] + self.last_dir_change_pos[1] * time_diff
                    self.eye_position = (int(pred_x), int(pred_y))
                    if self.logger:
                        self.logger.debug(f"使用预测位置: {self.eye_position}")

        # 计算移动方向
        move_direction = None
        if self.eye_position and self.last_eye_position:
            dx = self.eye_position[0] - self.last_eye_position[0]
            dy = self.eye_position[1] - self.last_eye_position[1]
            if abs(dx) > abs(dy):
                if dx > 2:
                    move_direction = 'right'
                elif dx < -2:
                    move_direction = 'left'
            else:
                if dy > 2:
                    move_direction = 'down'
                elif dy < -2:
                    move_direction = 'up'

            # 方向变化时记录位置和时间
            if move_direction and move_direction != self.head_direction:
                self.last_dir_change_pos = (dx, dy)
                self.last_dir_change_time = current_time

        # 用当前eye_position（无论新识别的还是缓存的）更新蛇头像素坐标
        if self.eye_position:
            avg_x, avg_y = self.eye_position
            self.head_position = (avg_x, avg_y)
            base_cell = self.board.get_cell_by_position(avg_x, avg_y)
            target_cell = None

            if base_cell:
                cx, cy = self.board.get_cell_center(base_cell.row, base_cell.col)
                offset_x = avg_x - cx
                offset_y = avg_y - cy

                border_threshold = 5

                if move_direction == 'right' and offset_x > border_threshold:
                    target_cell = self.board.get_cell_by_position(avg_x + 1, avg_y)
                elif move_direction == 'left' and offset_x < -border_threshold:
                    target_cell = self.board.get_cell_by_position(avg_x - 1, avg_y)
                elif move_direction == 'down' and offset_y > border_threshold:
                    target_cell = self.board.get_cell_by_position(avg_x, avg_y + 1)
                elif move_direction == 'up' and offset_y < -border_threshold:
                    target_cell = self.board.get_cell_by_position(avg_x, avg_y - 1)

                if target_cell:
                    self.remove_from_special_cells(target_cell.cell_type, target_cell)
                    target_cell.cell_type = "own_head"
                    self.add_to_special_cells("own_head", target_cell)
                else:
                    self.remove_from_special_cells(base_cell.cell_type, base_cell)
                    base_cell.cell_type = "own_head"
                    self.add_to_special_cells("own_head", base_cell)
            else:
                self.logger and self.logger.warning(f"蛇眼重心({avg_x},{avg_y})未匹配到格子")
        else:
            self.head_position = None  # 无缓存也没有新识别，置空

        self.determine_own_tail()
        self.last_frame = current_frame

        return True

    def analyze_color_move_direction(self, target_hsv=None, tolerance=[5,255,50], multi_frame=True, preset_direction=None):
        """
        根据颜色检测蛇头位置，并判断蛇头方向

        参数：
            target_hsv: list型，目标HSV颜色均值。例如[4,213,255], 默认为self.own_head色调
            tolerance: list型，HSV阈值上界范围，默认[5,50,50]
            multi_frame: bool，是否利用“上一帧”计算运动方向。默认True
            preset_direction: str，指定蛇头方向('up','down','left','right'或'none')，若提供则不进行多帧分析，直接使用

        返回：
            (方向字符串, 当前中心坐标元组, 蛇头边缘关键点坐标)
            如果检测失败，返回None
        """
        h_tol, s_tol, v_tol = tolerance
        if target_hsv is None:
            target_hsv = [self.own_head, 0, 0]

        target_h, target_s, target_v = target_hsv

        lower_hsv = np.array([
            max(0, target_h - h_tol),
            max(0, target_s - s_tol), 
            max(0, target_v - v_tol)
        ])
        upper_hsv = np.array([
            min(179, target_h + h_tol),
            min(255, target_s + s_tol),
            min(255, target_v + v_tol)
        ])

        current_mask = cv2.inRange(self.board.hsv_image, lower_hsv, upper_hsv)
        current_contours, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not current_contours:
            return None
        
        max_contour = max(current_contours, key=cv2.contourArea)
        M = cv2.moments(max_contour)

        if M["m00"] == 0:
            return None
        
        current_cx = int(M["m10"] / M["m00"])
        current_cy = int(M["m01"] / M["m00"])

        edge_point = (current_cx, current_cy)

        contour_points = max_contour.reshape(-1, 2)

        if preset_direction is not None:
            if preset_direction == 'all':
                edge_points = {}
                edge_points['right'] = tuple(contour_points[np.argmax(contour_points[:, 0])])
                edge_points['left'] = tuple(contour_points[np.argmin(contour_points[:, 0])])
                edge_points['down'] = tuple(contour_points[np.argmax(contour_points[:, 1])])
                edge_points['up'] = tuple(contour_points[np.argmin(contour_points[:, 1])])
                return ('all', (current_cx, current_cy), edge_points)
            else:
                direction = preset_direction
                if direction == 'right':
                    edge_point = tuple(contour_points[np.argmax(contour_points[:, 0])])
                elif direction == 'left':
                    edge_point = tuple(contour_points[np.argmin(contour_points[:, 0])])
                elif direction == 'down':
                    edge_point = tuple(contour_points[np.argmax(contour_points[:, 1])])
                elif direction == 'up':
                    edge_point = tuple(contour_points[np.argmin(contour_points[:, 1])])
                else:
                    edge_point = (current_cx, current_cy)
                return (direction, (current_cx, current_cy), edge_point)

        if not multi_frame or self.last_frame is None:
            direction = 'none'
            return (direction, (current_cx, current_cy), edge_point)
        
        last_mask = cv2.inRange(self.last_frame, lower_hsv, upper_hsv)
        last_contours, _ = cv2.findContours(last_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not last_contours:
            return ('none', (current_cx, current_cy), (current_cx, current_cy))

        last_max_contour = max(last_contours, key=cv2.contourArea)
        M_last = cv2.moments(last_max_contour)

        if M_last["m00"] == 0:
            return None

        last_cx = int(M_last["m10"] / M_last["m00"])
        last_cy = int(M_last["m01"] / M_last["m00"])

        dx = current_cx - last_cx
        dy = current_cy - last_cy

        direction = None
        if abs(dx) > abs(dy):
            direction = 'right' if dx > 0 else 'left'
        else:
            direction = 'down' if dy > 0 else 'up'

        if direction == 'right':
            edge_point = tuple(contour_points[np.argmax(contour_points[:, 0])])
        elif direction == 'left':
            edge_point = tuple(contour_points[np.argmin(contour_points[:, 0])])
        elif direction == 'down':
            edge_point = tuple(contour_points[np.argmax(contour_points[:, 1])])
        elif direction == 'up':
            edge_point = tuple(contour_points[np.argmin(contour_points[:, 1])])
        else:
            edge_point = (current_cx, current_cy)

        return (direction, (current_cx, current_cy), edge_point)

    def determine_precise_cell_position(self, direction, precise_pos):
        """
        :param direction: 移动方向 ('up','down','left','right')
        :param precise_pos: 精准坐标 (x,y)
        :return: [当前格子, 需要更新的格子] 或 None(当参数无效时)
        """
        if not direction or not precise_pos or not self.board:
            return None
            
        x, y = precise_pos
        
        # 获取当前格子
        current_cell = self.board.get_cell_by_position(x, y)
        if not current_cell:
            return None
            
        # 获取格子中心坐标
        cell_center = self.board.get_cell_center(current_cell.row, current_cell.col)
        if not cell_center:
            return [current_cell, None]
            
        cell_center_x, cell_center_y = cell_center
        
        # 获取当前格子的颜色分布字典
        color_dict = current_cell.get_color_dict(self.board.hsv_image, 3)
        head_color_count = color_dict.get(self.own_head, 0) if color_dict else 0
        total_pixels = sum(color_dict.values()) if color_dict else 1
        is_majority_head = (head_color_count / total_pixels) > 0.5
        
        # 根据方向判断是否需要更新到相邻格子
        new_cell = None
        if direction == 'right' and (x > cell_center_x or is_majority_head):
            new_cell = self.board.get_cell_by_position(x + 1, y)
        elif direction == 'left' and (x < cell_center_x or is_majority_head):
            new_cell = self.board.get_cell_by_position(x - 1, y)
        elif direction == 'down' and (y > cell_center_y or is_majority_head):
            new_cell = self.board.get_cell_by_position(x, y + 1)
        elif direction == 'up' and (y < cell_center_y or is_majority_head):
            new_cell = self.board.get_cell_by_position(x, y - 1)
            
        return [current_cell, new_cell]

    def find_snake_eye(self):
        """
        在整张图像中寻找纯白色蛇眼区域，返回蛇眼中心坐标列表 [(x1,y1),(x2,y2),...]
        """
        # 白色HSV阈值
        lower_white = np.array([0,0,255])
        upper_white = np.array([10,10,255])
        white_mask = cv2.inRange(self.board.hsv_image, lower_white, upper_white)

        # 找所有白色区域轮廓
        eye_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        eye_centers = []
        for cnt in eye_contours:
            if cv2.contourArea(cnt) < 0:  # 过滤特别小的噪声点
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            eye_centers.append((cx, cy))

        return eye_centers