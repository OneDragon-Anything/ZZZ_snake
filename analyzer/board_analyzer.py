import cv2
import numpy as np
import time
from pathlib import Path  # 需要新增这个导入
from log.debug_helper import DebugHelper
from model.image_cell import ImageCell
from model.template.template import Template
from model.template.template_manage import TemplateManager


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
            "own_head": [4, 213, 255],
            "own_body_tail": [15, 255, 255],
            "enemy_head": [127, 185, 255],  
            "enemy_body_tail": [130, 134, 255],
            "enemy2_head": [150, 250, 255],
            "enemy2_body_tail": [134, 134, 255],
            "enemy3_head": [17, 255, 255],
            "enemy3_body_tail": [30, 255, 255],
            "grid_light": [89, 80, 236],
            "grid_dark": [97, 251, 203],
            "game_over":  [109, 195, 88],
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
        self.grid_light = self.grid_colors_hsv["grid_light"][0]
        self.grid_dark = self.grid_colors_hsv["grid_dark"][0]
        self.game_over = self.grid_colors_hsv["game_over"][0]

        # 初始化模板管理器
        template_dir = Path(__file__).parent.parent / "templates"
        self.template_manager = TemplateManager(template_dir)

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

        # 进行深度分析
        self.deep_analysis()

        # 进行前后帧分析
        self.analyze_frame_changes(board_image)

        # 检测蛇头状态
        if "own_head" not in self.special_cells:
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
        """深度分析棋盘状态，包括：
        1. 初始化棋盘格子
        2. 继承上一帧的蛇身信息
        3. 全盘扫描识别特殊格子
        4. 模板匹配识别特殊元素
        5. 深度分析未知格子
        """
        if not self.board:
            return
        
        # 初始化所有格子为空白格
        for row in self.board.cells:
            for cell in row:
                cell.cell_type = "empty"
        
        self.special_cells = {}  # 重置特殊格子字典

        # 继承上一帧的蛇身信息（蛇头变为蛇身）
        previous = self.board.special_cells if hasattr(self.board, "special_cells") else {}
        inherited_body_cells = []

        # 处理所有蛇头类型，转换为蛇身
        for head_type in ["own_head", "enemy_head", "own_body", "enemy_body"]:
            for cell in previous.get(head_type, []):
                cell.cell_type = "own_body" if head_type in ["own_head", "own_body"] else "enemy_body"
                self.add_to_special_cells(cell.cell_type, cell)
                inherited_body_cells.append(cell)

        # 模板匹配识别特殊元素
        self.analyze_by_templates()

        # 全盘扫描识别特殊格子
        for row in self.board.cells:
            for cell in row:
                if cell.cell_type == "empty":  # 只处理未被继承的格子
                    cell_type = self.determine_cell_type(cell.center_color)
                    cell.cell_type = cell_type
                    if cell_type != "empty":
                        self.add_to_special_cells(cell_type, cell)

        # 深度分析未知格子
        unknow_cells = self.special_cells.get("unknow", [])
        if unknow_cells:
            for cell in unknow_cells[:]:  # 使用副本遍历
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
        if not self.board or not cell or not isinstance(cell, ImageCell):
            return None
            
        # 获取格子的颜色字典
        color_dict = cell.get_color_dict(self.board.hsv_image)
        # print(f"格子({cell.row+1}, {cell.col+1})的颜色分布: {color_dict}")
        
        if not color_dict:
            return cell.cell_type
            
        # 按颜色出现频率从高到低排序
        sorted_colors = sorted(color_dict.items(), key=lambda x: x[1], reverse=True)
        
        # 遍历所有颜色，找到第一个确定的类型
        for h_value, count in sorted_colors:
            # 构造HSV颜色值进行类型判断
            new_type = self.determine_cell_type((h_value, 170, 255))
            # print(f"分析颜色H={h_value}, 数量={count}, 判定类型={new_type}")
            
            # 如果找到非empty和非unknow的类型
            if new_type not in ["empty", "unknow"]:
                # 从原类型列表中移除该格子
                self.remove_from_special_cells(cell.cell_type, cell)
                # 更新格子的中心颜色和类型
                cell.center_color = (h_value, 0, 0)
                cell.cell_type = new_type
                # 添加到新类型列表中
                self.add_to_special_cells(new_type, cell)
                # print(f"格子({cell.row+1}, {cell.col+1})更新为: {new_type}")
                return new_type
        
        # 如果所有颜色都是empty或unknow，保持原状
        # print(f"格子({cell.row+1}, {cell.col+1})保持原状: {cell.cell_type}")
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
            color_dict = cell.get_color_dict(self.board.hsv_image)
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
        if v >=250 :
            if self.own_head <= h <= self.own_tail:
                return "own_body"
            elif h == self.enemy_head:
                return "enemy_head"
            elif h == self.enemy2_head:
                return "enemy_head"
            elif h == self.enemy3_head:
                return "enemy_head"
            elif (self.enemy2_tail <= h <= self.enemy2_head) or (self.enemy_tail >= h >= self.enemy_head) or (self.enemy3_head <= h <= self.enemy3_tail):
                return "enemy_body"
        # 检查是否在空格子的HSV范围内

        h_in_range = self.grid_light <= h <= self.grid_dark
        s_in_range = self.grid_colors_hsv["grid_light"][1] <= s <= self.grid_colors_hsv["grid_dark"][1]
        if h_in_range and s_in_range:
            return "empty"

        return "unknow"
            
    def add_to_special_cells(self, cell_type, cell, special_cells=None):
        """
        将单元格添加到special_cells字典中，避免坐标重复
        :param cell_type: 单元格类型
        :param cell: 要添加的单元格对象
        :param special_cells: 可选，指定要操作的特殊格字典，None则使用self.special_cells
        """
        if cell is None:
            return  # 防止None被添加进来

        # 可选！严格类型判断
        if not isinstance(cell, ImageCell):
            return  # 不是合法棋盘格，跳过

        target_cells = special_cells if special_cells is not None else self.special_cells
        
        if cell_type not in target_cells:
            target_cells[cell_type] = []

        # 先移除坐标相同的已有cell
        target_cells[cell_type] = [
            c for c in target_cells[cell_type]
            if not (c.row == cell.row and c.col == cell.col)
        ]

        # 添加新cell（坐标唯一）
        target_cells[cell_type].append(cell)

    def remove_from_special_cells(self, cell_type, cell, special_cells=None):
        """
        从special_cells字典中移除单元格
        :param cell_type: 单元格类型
        :param cell: 要移除的单元格对象
        :param special_cells: 可选，指定要操作的特殊格字典，None则使用self.special_cells
        """
        target_cells = special_cells if special_cells is not None else self.special_cells
        if cell_type in target_cells and cell in target_cells[cell_type]:
            target_cells[cell_type].remove(cell)
        
    def analyze_frame_changes(self, current_frame):
        """
        分析前后帧的变化，并根据蛇眼判断蛇头格子

        :param current_frame: 当前帧HSV图像
        :return: True
        """
        current_time = time.time()
        eyes = self.find_snake_eye()
        if not eyes:
            return False
        avg_x = int(np.mean([p[0] for p in eyes]))
        avg_y = int(np.mean([p[1] for p in eyes]))
        self.eye_position = (avg_x, avg_y)

        # 用当前eye_position（无论新识别的还是缓存的）更新蛇头像素坐标
        if self.eye_position:
            avg_x, avg_y = self.eye_position
            self.head_position = (avg_x, avg_y)
            base_cell = self.board.get_cell_by_position(avg_x, avg_y)
            target_cell = None

            if base_cell: 
                head_x, head_y = base_cell.col, base_cell.row
                best_cell = None #最佳格
                max_count = 0 #合法最大计数
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue  # 跳过中心点(蛇头本身)
                        nx, ny = head_x + dx, head_y + dy
                        if 0 <= nx < self.board.cols and 0 <= ny < self.board.rows:
                            neighbor = self.board.cells[ny][nx]
                            if neighbor.cell_type == "empty":
                                self.deep_analysis_cell(neighbor)
                                color_dict = neighbor.get_color_dict(self.board.hsv_image)
                                has_head_color = 4 in [int(h_value) for h_value in color_dict.keys()]
                                if has_head_color:
                                    count = sum(cnt for h_value, cnt in color_dict.items() 
                                              if self.grid_light <= int(h_value) <= self.grid_dark)
                                    if count > max_count:
                                        max_count = count
                                        best_cell = neighbor

                # 确定最终的蛇头cell
                head_cell = best_cell if best_cell else base_cell
                
                # 更新蛇头cell类型
                self.remove_from_special_cells(head_cell.cell_type, head_cell)
                head_cell.cell_type = "own_head"
                self.add_to_special_cells("own_head", head_cell)

        else:
            self.head_position = None  # 无缓存也没有新识别，置空

        self.determine_own_tail()
        self.last_frame = current_frame


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
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            eye_centers.append((cx, cy))

        return eye_centers

    def analyze_by_templates(self):
        """
        使用模板管理器分析棋盘并更新格子类型
        返回:
            更新后的special_cells字典
        """
        # 如果已有模板则直接使用，否则获取新模板
        if not hasattr(self, '_cached_templates'):
            self.logger and self.logger.debug("获取新模板")
            self._cached_templates = self.template_manager.get_all_templates()
            
        # 获取所有snake模板
        snake_templates = [
            template for template in self._cached_templates
            if template.frame_id.lower() == "snake"
        ]
        
        if not snake_templates or not self.board:
            return self.special_cells
        
        # 逐个模板匹配
        for template in snake_templates:
            # 如果模板ID包含"蛇"则跳过
            if "snake" in template.template_id:
                continue
                
            if self.logger:
                has_image = hasattr(template, 'image_path') and template.image_path
            
            matches = self.template_manager.find_objects_by_features(
                template, 
                hsv_image=self.board.hsv_image
            )
        

            if not matches:
                continue
            # 根据模板ID确定格子类型
            if template.template_id.lower() in ["diamond", "yellow_crystal"]:
                cell_type = "score_boost"
            else:
                cell_type = template.template_id.lower()  # 默认类型
            # 将匹配结果映射到棋盘格子
            for x, y, angle, score in matches:
                
                if template.template_id.lower() in ["mine", "super_star"]:
                    # 直接获取四个方向的格子
                    positions = [
                        (x+5, y+5),     # 右
                        (x-5, y-5),     # 左 
                        (x-5, y+5),     # 下
                        (x+5, y-5)      # 上
                    ]
                    for px, py in positions:
                        cell = self.board.get_cell_by_position(px, py)
                        if cell:
                            self.remove_from_special_cells(cell.cell_type, cell)
                            cell.cell_type = cell_type
                            self.add_to_special_cells(cell_type, cell)
                else:
                    # 普通处理方式
                    cell = self.board.get_cell_by_position(x, y)
                    if cell:
                        self.remove_from_special_cells(cell.cell_type, cell)
                        cell.cell_type = cell_type
                        self.add_to_special_cells(cell_type, cell)


        
        return self.special_cells