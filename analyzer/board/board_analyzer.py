import cv2
import numpy as np
import time
from pathlib import Path
from model.image_cell import ImageCell
from analyzer.board.color_analyzer import ColorAnalyzer
from analyzer.board.template_analyzer import TemplateAnalyzer
from analyzer.board.config.color_config import ColorConfig


class BoardAnalyzer:
    """负责分析棋盘状态的类 - 重构版本"""

    def __init__(self, logger=None):
        self.logger = logger
        self.board = None
        self.special_cells = {}
        self.special_cells_coords = {}  # 使用字典存储坐标集合，提高查找效率
        
        # 状态变量
        self.head_position = None  # 蛇头精准坐标 (x,y)
        self.head_direction = None  # 蛇头运动方向 ('up','down','left','right')
        self.is_gameover = False
        self.is_running = False
        self.last_gameover_check = 0  # 上次检测gameover的时间戳
        
        # 眼睛跟踪相关
        self.last_frame = None  # 上一帧画面
        self.last_target_pos = []  # 存储多个终点关注单位的坐标
        self.last_no_head_save_time = 0
        self.last_eye_position = None  # 上一帧蛇眼重心坐标
        self.eye_position = None  # 当前帧蛇眼重心坐标
        self.last_eye_time = 0  # 最后检测到眼睛的时间戳
        self.last_dir_change_pos = None  # 最后方向变化时的位置差(dx,dy)
        self.last_dir_change_time = 0  # 最后方向变化的时间戳
        self.eye_velocity = (0, 0)  # 蛇眼移动速度 (vx,vy) 像素/秒

        # 初始化分析器组件
        self.color_analyzer = ColorAnalyzer()
        self.template_analyzer = TemplateAnalyzer()
        
        # 颜色常量
        self.game_over = ColorConfig.get_h_value("game_over")
        self.own_tail = ColorConfig.get_h_value("own_body_tail")

    def analyze_board(self, board, board_image=None, image_format="HSV", last_key_direction=None):
        """分析棋盘图像并初始化棋盘状态
        :param board: 棋盘对象
        :param board_image: 输入的棋盘图像
        :param image_format: 图像格式，默认HSV
        :param last_key_direction: 上一次按键方向
        :return: 更新后的Board对象
        """
        start_time = time.time()
        self.head_direction = last_key_direction
        
        # 设置图像
        if board_image is not None:
            board.set_hsv_image(board_image, image_format)

        # 存储棋盘对象
        self.board = board

        # 检测gameover状态（每10秒检测一次）
        self._check_gameover_status()
        if self.is_gameover:
            return board

        # 进行深度分析
        self._deep_analysis()

        # 进行前后帧分析
        self._analyze_frame_changes(board_image)

        # 清理空列表
        self.special_cells = {k: v for k, v in self.special_cells.items() if v}

        # 检查未知格子数量
        unknown_count = len(self.special_cells.get("unknown", []))
        if unknown_count > 10:
            self.is_running = False
            self.head_position = None
            self.head_direction = None
            self.last_frame = None
            return board
        else:
            self.is_running = True

            # 更新棋盘对象
            board.special_cells = self.special_cells
            board.head_position = self.head_position
            board.direction = self.head_direction
            
            return board

    def _check_gameover_status(self):
        """检查游戏是否结束"""
        current_time = time.time()
        if current_time - self.last_gameover_check >= 3:
            # 将棋盘转换为NumPy数组
            cells = np.array(self.board.cells)
            
            # 提取所有格子的H值
            h_values = np.array([[cell.center_color[0] for cell in row] for row in cells])
            
            # 向量化计算gameover色匹配的格子数量
            gameover_count = np.sum(np.abs(h_values - self.game_over) <= 1)
            
            if 88 <= gameover_count <= 92:
                self.is_gameover = True
                return

            self.last_gameover_check = current_time
            self.is_gameover = False

    def _deep_analysis(self):
        """深度分析棋盘状态，包括：
        1. 初始化棋盘格子
        2. 继承上一帧的蛇身信息
        3. 全盘扫描识别特殊格子
        4. 模板匹配识别特殊元素
        5. 深度分析未知格子
        """
        if not self.board:
            return

        # 使用numpy向量化操作初始化所有格子
        cells = np.array(self.board.cells)
        for cell in cells.flat:
            h, s, _ = cell.center_color
            h_in_range = self.color_analyzer.grid_light <= h <= self.color_analyzer.grid_dark
            s_in_range = (
                ColorConfig.DEFAULT_COLORS["grid_light"][1]
                <= s
                <= ColorConfig.DEFAULT_COLORS["grid_dark"][1]
            )
            cell.cell_type = "empty" if h_in_range and s_in_range else "unknown"

        # 重置特殊格子字典和坐标集合
        self.special_cells = {}
        self.special_cells_coords = {}

        # 继承上一帧的蛇身信息（蛇头变为蛇身）
        self._inherit_previous_frame_info()
        
        # 模板匹配识别特殊元素
        
        self._analyze_by_templates()

        # 向量化处理空白和未知格子的类型判断
        self._analyze_remaining_cells(cells)

    def _inherit_previous_frame_info(self):
        """继承上一帧的蛇身信息"""
        previous = self.board.special_cells if hasattr(self.board, "special_cells") else {}
        
        # 批量处理蛇头类型转换为蛇身
        head_types = {"own_head": "own_body", "enemy_head": "enemy_body", 
                     "own_body": "own_body", "enemy_body": "enemy_body"}
        for head_type, body_type in head_types.items():
            if head_type in previous:
                for cell in previous[head_type]:
                    cell.cell_type = body_type
                    self.add_to_special_cells(body_type, cell)

    def _analyze_by_templates(self):
        """使用模板分析器识别特殊元素"""
        # 获取模板匹配结果
        all_matches = self.template_analyzer.analyze_by_templates(self.board)
        
        # 更新特殊格子
        for category, matches in all_matches.items():
            cells_to_update = self.template_analyzer.get_cells_to_update(self.board, matches, category)
            for cell in cells_to_update:
                self.remove_from_special_cells(cell.cell_type, cell)
                cell.cell_type = category
                self.add_to_special_cells(category, cell)

    def _analyze_remaining_cells(self, cells):
        """分析剩余的空白和未知格子"""
        cells_to_analyze = [cell for cell in cells.flat if cell.cell_type in ["empty", "unknown"]]
        if not cells_to_analyze:
            return
            
        # 批量获取中心颜色并判断类型
        for cell in cells_to_analyze:
            cell_type = self.color_analyzer.determine_cell_type(cell.center_color)
            if cell_type == "unknown":
                cell_type = self.color_analyzer.deep_analysis_cell(cell, self.board.hsv_image)
            cell.cell_type = cell_type
            if cell_type != "empty":
                self.add_to_special_cells(cell_type, cell)

    def _analyze_frame_changes(self, current_frame):
        """分析前后帧的变化，并根据蛇眼判断蛇头格子
        :param current_frame: 当前帧HSV图像
        """
        # 查找蛇眼
        eyes = self.color_analyzer.find_snake_eye(self.board.hsv_image)
        if eyes:
            avg_x = int(np.mean([p[0] for p in eyes]))
            avg_y = int(np.mean([p[1] for p in eyes]))
            self.eye_position = (avg_x, avg_y)

            # 用当前eye_position更新蛇头像素坐标
            self.head_position = (avg_x, avg_y)
            base_cell = self.board.get_cell_by_position(avg_x, avg_y)
            
            if base_cell:
                head_cell = self._determine_head_cell(base_cell)
                # self._deep_analysis_head_neighbors(head_cell)
        else:
            self.head_position = None  # 无缓存也没有新识别，置空

        # 确定蛇尾
        self._determine_own_tail()

        # 确定敌人蛇,暂不需要
        # self._determine_enemy_snake()
        
        # 更新上一帧
        self.color_analyzer.update_last_frame(current_frame)
        self.last_frame = current_frame

    def _determine_head_cell(self, base_cell):
        """优化后的蛇头格子确定方法"""
        head_x, head_y = base_cell.col, base_cell.row
        best_cell = None
        max_count = 0
        
        # 预计算颜色范围
        head_h_min = self.color_analyzer.own_head
        head_h_max = self.color_analyzer.own_body
        grid_h_min = self.color_analyzer.grid_light
        grid_h_max = self.color_analyzer.grid_dark

        # 检查周围4个格子
        for dx, dy in [(-1,0), (0,-1), (0,1), (1,0)]:
            nx, ny = head_x + dx, head_y + dy
            if 0 <= nx < self.board.cols and 0 <= ny < self.board.rows:
                neighbor = self.board.cells[ny][nx]
                
                # 只看空格
                if not neighbor.cell_type in ["empty"]:
                    continue
                    
                # 分析格子颜色
                color_dict = neighbor.get_color_dict(self.board.hsv_image)
                
                # 快速统计匹配像素
                head_count = 0
                grid_count = 0
                for h_value, cnt in color_dict.items():
                    h_int = int(h_value)
                    if head_h_min <= h_int <= head_h_max:
                        head_count += cnt
                    if grid_h_min <= h_int <= grid_h_max:
                        grid_count += cnt
                
                # 只有当有蛇头颜色时才考虑
                if head_count > 0 and grid_count > max_count:
                    max_count = grid_count
                    best_cell = neighbor

        # 确定最终蛇头格子
        head_cell = best_cell if best_cell and max_count > 1 else base_cell
        
        # 更新格子类型
        self.remove_from_special_cells(head_cell.cell_type, head_cell)
        head_cell.cell_type = "own_head"
        self.add_to_special_cells("own_head", head_cell)
        return head_cell

    def _determine_own_tail(self):
        """简化版蛇尾检测算法"""
        if "own_body" not in self.special_cells:
            return None

        # 第一步：找出中心色最接近蛇尾的own_body格子
        best_cell = None
        min_diff = float('inf')
        
        for cell in self.special_cells["own_body"]:
            # 直接使用中心色进行比较
            h_value = cell.center_color[0]
            diff = abs(int(h_value) - self.own_tail)
            
            if diff < min_diff:
                min_diff = diff
                best_cell = cell

        if not best_cell:
            return None

        # 第二步：生成候选格子列表（包括自身和相邻格子）
        candidates = []
        x, y = best_cell.col, best_cell.row
        
        # 添加自身和四个相邻格子
        candidates.append(best_cell)
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board.cols and 0 <= ny < self.board.rows:
                if self.board.cells[ny][nx].cell_type == "empty":
                    candidates.append(self.board.cells[ny][nx])

        # 筛选条件：有蛇尾颜色且有空格色
        for cell in candidates:
            color_dict = cell.get_color_dict(self.board.hsv_image)
            if not color_dict:
                continue
                
            # 检查是否有蛇尾颜色
            has_tail_color = any(abs(int(h) - self.own_tail) <= 1 for h in color_dict)
            # 检查是否有空格色
            has_grid_color = any(
                self.color_analyzer.grid_light <= int(h) <= self.color_analyzer.grid_dark
                for h in color_dict
            )
            
            if has_tail_color and has_grid_color:
                self._update_tail_cell(cell)
                return cell

        # 如果没有找到符合条件的格子，返回中心色最接近的格子
        if best_cell:
            self._update_tail_cell(best_cell)
            return best_cell

        return None

    def _update_tail_cell(self, cell):
        """更新格子为蛇尾的通用方法"""
        self.remove_from_special_cells(cell.cell_type, cell)
        cell.cell_type = "own_tail"
        self.add_to_special_cells("own_tail", cell)

    def _deep_analysis_head_neighbors(self, head_cell):
        """深度分析蛇头周围3x3的空白格子"""
        if not head_cell or not self.board:
            return
    
        head_x, head_y = head_cell.col, head_cell.row
        
        # 检查3x3范围内的格子
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                # 跳过中心格子(蛇头本身)
                if dx == 0 and dy == 0:
                    continue
                    
                nx, ny = head_x + dx, head_y + dy
                if 0 <= nx < self.board.cols and 0 <= ny < self.board.rows:
                    neighbor = self.board.cells[ny][nx]
                    
                    # 只分析空白格子
                    if neighbor.cell_type == "empty":
                        # 深度分析该格子
                        cell_type = self.color_analyzer.deep_analysis_cell(
                            neighbor, 
                            self.board.hsv_image
                        )
                        
                        # 更新格子类型
                        if cell_type != "empty":
                            self.remove_from_special_cells(neighbor.cell_type, neighbor)
                            neighbor.cell_type = cell_type
                            self.add_to_special_cells(cell_type, neighbor)

    def _determine_enemy_snake(self):
        """处理敌方小蛇蛇头蛇尾"""
        if "enemy_head" not in self.special_cells and "enemy_body" not in self.special_cells:
            return None

        # 1. 收集所有敌方蛇身和蛇头
        enemy_cells = []
        if "enemy_head" in self.special_cells:
            enemy_cells.extend(self.special_cells["enemy_head"])
        
        if not enemy_cells:
            return None

        # 2. 分组敌方蛇身(基于连通性)
        groups = self._group_enemy_cells(enemy_cells)

        # 3. 输出每组蛇头坐标
        for i, group in enumerate(groups):
            head_cells = [cell for cell in group if cell.cell_type == "enemy_head"]
            if head_cells:
                if self.logger:
                    coords = [(cell.col+1, cell.row+1) for cell in head_cells]
                    self.logger.debug(f"敌方蛇组{i+1} 蛇头坐标: {coords}")

    def _group_enemy_cells(self, cells):
        """基于连通性分组敌方蛇身"""
        # 使用并查集算法分组
        parent = {}
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
            
        def union(u, v):
            u_root = find(u)
            v_root = find(v)
            if u_root == v_root:
                return
            parent[u_root] = v_root
            
        # 初始化并查集
        for cell in cells:
            parent[(cell.row, cell.col)] = (cell.row, cell.col)
        
        # 连接相邻的敌方格子
        for cell in cells:
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = cell.col + dx, cell.row + dy
                if 0 <= nx < self.board.cols and 0 <= ny < self.board.rows:
                    neighbor = self.board.cells[ny][nx]
                    if neighbor in cells:
                        union((cell.row, cell.col), (neighbor.row, neighbor.col))
        
        # 获取最终分组
        groups = {}
        for cell in cells:
            root = find((cell.row, cell.col))
            if root not in groups:
                groups[root] = []
            groups[root].append(cell)
            
        return list(groups.values())


    def add_to_special_cells(self, cell_type, cell):
        """将单元格添加到special_cells字典中，避免坐标重复
        :param cell_type: 单元格类型
        :param cell: 要添加的单元格对象
        """
        if cell is None or not isinstance(cell, ImageCell):
            return
        
        # 初始化类型对应的列表和坐标集合
        if cell_type not in self.special_cells:
            self.special_cells[cell_type] = []
            self.special_cells_coords[cell_type] = set()

        coords = (cell.row, cell.col)
        coords_set = self.special_cells_coords[cell_type]
        
        # 如果坐标已存在，先移除旧的cell
        if coords in coords_set:
            self.special_cells[cell_type] = [c for c in self.special_cells[cell_type] if (c.row, c.col) != coords]
        else:
            coords_set.add(coords)
            
        # 添加新cell
        self.special_cells[cell_type].append(cell)

    def remove_from_special_cells(self, cell_type, cell):
        """从special_cells字典中移除单元格
        :param cell_type: 单元格类型
        :param cell: 要移除的单元格对象
        """
        if cell_type in self.special_cells and cell in self.special_cells[cell_type]:
            self.special_cells[cell_type].remove(cell)
            if hasattr(cell, 'row') and hasattr(cell, 'col'):
                coords = (cell.row, cell.col)
                if cell_type in self.special_cells_coords:
                    self.special_cells_coords[cell_type].discard(coords)