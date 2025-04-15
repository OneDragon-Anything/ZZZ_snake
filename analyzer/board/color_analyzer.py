import cv2
import numpy as np
import time
from model.image_cell import ImageCell
from analyzer.board.config.color_config import ColorConfig


class ColorAnalyzer:
    """负责颜色分析的类，从BoardAnalyzer中分离出来"""

    def __init__(self):
        # 初始化颜色参数
        self.grid_colors_hsv = ColorConfig.DEFAULT_COLORS
        
        # 预设颜色参数
        self.own_head = ColorConfig.get_h_value("own_head")
        self.own_body = ColorConfig.get_h_value("own_body")
        self.own_tail = ColorConfig.get_h_value("own_body_tail")
        self.enemy_head = ColorConfig.get_h_value("enemy_head")
        self.enemy_tail = ColorConfig.get_h_value("enemy_body_tail")
        self.enemy2_head = ColorConfig.get_h_value("enemy2_head")
        self.enemy2_tail = ColorConfig.get_h_value("enemy2_body_tail")
        self.enemy3_head = ColorConfig.get_h_value("enemy3_head")
        self.enemy3_tail = ColorConfig.get_h_value("enemy3_body_tail")
        self.enemy4_head = ColorConfig.get_h_value("enemy4_head")
        self.enemy4_body = ColorConfig.get_h_value("enemy4_body")
        self.enemy4_body_tail = ColorConfig.get_h_value("enemy4_body_tail")
        self.grid_light = ColorConfig.get_h_value("grid_light")
        self.grid_dark = ColorConfig.get_h_value("grid_dark")
        self.game_over = ColorConfig.get_h_value("game_over")
        
        # 缓存上一帧
        self.last_frame = None

    def determine_cell_type(self, hsv_color):
        """根据HSV颜色确定格子类型
        :param hsv_color: HSV颜色值
        :return: 格子类型
        """
        h, s, v = hsv_color

        # 判断各种类型
        if v >= 250:
            if self.own_head <= h <= self.own_tail:
                return "own_body"
            elif h == self.enemy_head:
                return "enemy_head"
            elif h == self.enemy2_head:
                return "enemy_head"
            elif h == self.enemy3_head:
                return "enemy_head"
            elif h == self.enemy4_head:
                return "enemy_head"
            elif (
                (self.enemy2_tail <= h <= self.enemy2_head)
                or (self.enemy_tail >= h >= self.enemy_head)
                or (self.enemy3_head <= h <= self.enemy3_tail)
                or (self.enemy4_body_tail <= h <= self.enemy4_body)
            ):
                return "enemy_body"
        
        # 检查是否在空格子的HSV范围内
        h_in_range = self.grid_light <= h <= self.grid_dark
        s_in_range = (
            self.grid_colors_hsv["grid_light"][1]
            <= s
            <= self.grid_colors_hsv["grid_dark"][1]
        )
        if h_in_range and s_in_range:
            return "empty"

        return "unknown"

    def deep_analysis_cell(self, cell, hsv_image):
        """深度分析指定的单个格子
        功能：对单个格子进行更精确的颜色分析，根据颜色分布确定格子类型
        参数：
            cell: 要分析的格子对象
            hsv_image: HSV格式的图像
        返回：
            更新后的格子类型字符串
        """
        # 检查输入参数有效性
        if hsv_image is None or not cell or not isinstance(cell, ImageCell):
            return None

        # 获取格子的颜色字典
        color_dict = cell.get_color_dict(hsv_image)

        if not color_dict:
            return cell.cell_type

        # 过滤掉出现次数少于26的颜色
        filtered_colors = {k:v for k,v in color_dict.items() if v >= 26}
        # 按颜色出现频率从高到低排序
        sorted_colors = sorted(filtered_colors.items(), key=lambda x: x[1], reverse=True)

        # 遍历所有颜色，找到第一个确定的类型
        for h_value, count in sorted_colors:
            # 构造HSV颜色值进行类型判断
            new_type = self.determine_cell_type((h_value, 170, 255))

            # 如果找到非empty和非unknown的类型
            if new_type not in ["empty", "unknown"]:
                # 更新格子的中心颜色
                cell.center_color = (h_value, 0, 0)
                return new_type

        # 如果所有颜色都是empty或unknown，保持原状
        return cell.cell_type

    def find_snake_eye(self, hsv_image):
        """在整张图像中寻找纯白色蛇眼区域，返回蛇眼中心坐标列表 [(x1,y1),(x2,y2),...]"""
        # 白色HSV阈值
        lower_white = np.array([0, 0, 255])
        upper_white = np.array([10, 10, 255])
        white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

        # 找所有白色区域轮廓
        eye_contours, _ = cv2.findContours(
            white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        eye_centers = []
        for cnt in eye_contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            eye_centers.append((cx, cy))

        return eye_centers
    
    def update_last_frame(self, current_frame):
        """更新上一帧图像"""
        self.last_frame = current_frame.copy() if current_frame is not None else None