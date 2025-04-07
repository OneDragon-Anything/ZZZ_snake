import cv2
import numpy as np
import time
from log import SnakeLogger

class ImageAnalyzer:
    """负责分析图像生成地图的类"""
    def __init__(self, logger=None):
        self.logger = logger
        self.game_state = "initial"  # 初始状态
        
        # 定义颜色阈值（HSV格式）
        self.grid_colors_hsv = {
            "speed_boost": [72, 149, 235],
            "score_boost": [22, 50, 255],
            "own_head": [4, 213, 255],
            "own_body_tail": [15, 255, 255],
            "enemy_head": [127, 185, 255],
            "enemy_body_tail": [130, 134, 255],
            "grid_light": [97, 85, 221],
            "grid_dark": [94, 247, 203],
            "game_over":  [109, 192, 88],
        }
        
        # 预设颜色参数
        self.own_head = self.grid_colors_hsv["own_head"][0]
        self.own_tail = self.grid_colors_hsv["own_body_tail"][0]
        self.enemy_head = self.grid_colors_hsv["enemy_head"][0]
        self.enemy_tail = self.grid_colors_hsv["enemy_body_tail"][0]
        self.speed_boost = self.grid_colors_hsv["speed_boost"][0]
        self.score_boost = self.grid_colors_hsv["score_boost"][0]
        self.grid_light = self.grid_colors_hsv["grid_light"][0]
        self.grid_dark = self.grid_colors_hsv["grid_dark"][0]
        self.game_over = self.grid_colors_hsv["game_over"][0]
    
    def analyze_frame(self, frame):
        """
        分析游戏帧，更新游戏状态
        :param frame: OpenCV格式的游戏画面
        :return: (game_state, board_state, special_cells)
        """
        board_state = self.analyze_board(frame)
        special_cells = {}
        has_snake_head = False
        game_over = False
        
        # 分析棋盘状态并记录特殊格子
        for y in range(len(board_state)):
            for x in range(len(board_state[y])):
                cell = board_state[y][x]
                if cell == "game_over":
                    game_over = True
                    break
                if cell == "own_head":
                    has_snake_head = True
                if cell not in ["empty"]:
                    special_cells[(x, y)] = cell
        
        # 根据是否有蛇头更新游戏状态
        if has_snake_head:
            self.game_state = "running"
        elif game_over:
            self.game_state = "game_over"
        else:
            self.game_state = "initial"
            
        return self.game_state, board_state, special_cells
    
    def analyze_board(self, frame):
        """
        分析游戏棋盘
        :param frame: 游戏画面
        :return: 棋盘状态二维数组
        """
        try:
            # 获取图像尺寸
            h, w = frame.shape[:2]

            # 定义棋盘大小
            grid_h, grid_w = 25, 29
            
            # 计算每个格子的大小
            cell_h = h / grid_h
            cell_w = w / grid_w

            # 创建棋盘状态数组，默认所有格子为empty
            board_state = [["empty" for _ in range(grid_w)] for _ in range(grid_h)]
            
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 分析每个格子
            for r in range(grid_h):
                for c in range(grid_w):
                        
                    # 计算格子中心点
                    center_x = int(c * cell_w + cell_w / 2)
                    center_y = int(r * cell_h + cell_h / 2)
                    
                    # 获取中心点颜色
                    if 0 <= center_y < h and 0 <= center_x < w:
                        cell_color = hsv[center_y, center_x]
                        
                        # 根据颜色判断格子类型
                        cell_type = self.determine_cell_type(cell_color)
                        if cell_type is not None:
                            board_state[r][c] = cell_type
            
            return board_state
        except Exception as e:
            if self.logger:
                self.logger.log(f"分析棋盘出错: {str(e)}")
            return [[None for _ in range(29)] for _ in range(25)]
    
    def determine_cell_type(self, hsv_color):
        """
        根据HSV颜色确定格子类型
        :param hsv_color: HSV颜色值
        :return: 格子类型
        """
        h, s, v = hsv_color
        
        # 判断各种类型
        if h == self.own_head:
            return "own_head"
        elif min(self.own_head, self.own_tail) < h < max(self.own_head, self.own_tail):
            return "own_body"
        elif h == self.enemy_head:
            return "enemy_head"
        elif min(self.enemy_head, self.enemy_tail) < h < max(self.enemy_head, self.enemy_tail):
            return "enemy_body"
        elif abs(h - self.speed_boost) <= 5:
            return "speed_boost"
        elif abs(h - self.score_boost) <= 5:
            return "score_boost"
        elif h == self.game_over:
            return "game_over"
        return None

    # 分析蛇头位置，返回分析后的蛇头坐标
    def find_similar_color_center(self, hsv_frame, center_x, center_y, side_length, target_hue=None, hue_tolerance=0):
        """
        以给定坐标为中心，在指定范围内检测同色调像素，并计算它们的中心坐标。

        Args:
            hsv_frame: 要检测的HSV格式图像
            center_x: 起始坐标的 x 值.
            center_y: 起始坐标的 y 值.
            side_length: 检测范围的边长.
            target_hue: 目标色调值(可选，如果不提供则使用中心点色调)
            hue_tolerance: 色调的允许误差范围.

        Returns:
            同色调像素的中心坐标 (x, y)，如果没有找到则返回 None.
        """
        frame_height, frame_width = hsv_frame.shape[:2]

        # 1. 获取起始坐标色调
        initial_hue = target_hue if target_hue is not None else hsv_frame[center_y, center_x, 0]

        # 2. 确定检测范围
        start_x = max(0, center_x - side_length // 2)
        end_x = min(frame_width, center_x + side_length // 2)
        start_y = max(0, center_y - side_length // 2)
        end_y = min(frame_height, center_y + side_length // 2)

        similar_pixels = []

        # 3. 检测同色调像素
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                hue = hsv_frame[y, x, 0]
                if abs(int(hue) - int(initial_hue)) <= hue_tolerance:
                    similar_pixels.append((x, y))

        # 4. 计算中心坐标
        if similar_pixels:
            x_coords = [p[0] for p in similar_pixels]
            y_coords = [p[1] for p in similar_pixels]
            avg_x = np.mean(x_coords)
            avg_y = np.mean(y_coords)
            return avg_x, avg_y
        else:
            return None
