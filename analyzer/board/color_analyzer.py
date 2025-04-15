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

        # 按颜色出现频率从高到低排序
        sorted_colors = sorted(color_dict.items(), key=lambda x: x[1], reverse=True)

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

    def analyze_color_move_direction(
        self,
        hsv_image,
        target_hsv=None,
        tolerance=[5, 255, 50],
        multi_frame=True,
        preset_direction=None,
    ):
        """根据颜色检测蛇头位置，并判断蛇头方向

        参数：
            hsv_image: HSV格式的图像
            target_hsv: list型，目标HSV颜色均值。例如[4,213,255], 默认为self.own_head色调
            tolerance: list型，HSV阈值上界范围，默认[5,50,50]
            multi_frame: bool，是否利用"上一帧"计算运动方向。默认True
            preset_direction: str，指定蛇头方向('up','down','left','right'或'none')，若提供则不进行多帧分析，直接使用

        返回：
            (方向字符串, 当前中心坐标元组, 蛇头边缘关键点坐标)
            如果检测失败，返回None
        """
        h_tol, s_tol, v_tol = tolerance
        if target_hsv is None:
            target_hsv = [self.own_head, 0, 0]

        target_h, target_s, target_v = target_hsv

        lower_hsv = np.array(
            [
                max(0, target_h - h_tol),
                max(0, target_s - s_tol),
                max(0, target_v - v_tol),
            ]
        )
        upper_hsv = np.array(
            [
                min(179, target_h + h_tol),
                min(255, target_s + s_tol),
                min(255, target_v + v_tol),
            ]
        )

        current_mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
        current_contours, _ = cv2.findContours(
            current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

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
            if preset_direction == "all":
                edge_points = {}
                edge_points["right"] = tuple(
                    contour_points[np.argmax(contour_points[:, 0])]
                )
                edge_points["left"] = tuple(
                    contour_points[np.argmin(contour_points[:, 0])]
                )
                edge_points["down"] = tuple(
                    contour_points[np.argmax(contour_points[:, 1])]
                )
                edge_points["up"] = tuple(
                    contour_points[np.argmin(contour_points[:, 1])]
                )
                return ("all", (current_cx, current_cy), edge_points)
            else:
                direction = preset_direction
                if direction == "right":
                    edge_point = tuple(contour_points[np.argmax(contour_points[:, 0])])
                elif direction == "left":
                    edge_point = tuple(contour_points[np.argmin(contour_points[:, 0])])
                elif direction == "down":
                    edge_point = tuple(contour_points[np.argmax(contour_points[:, 1])])
                elif direction == "up":
                    edge_point = tuple(contour_points[np.argmin(contour_points[:, 1])])
                else:
                    edge_point = (current_cx, current_cy)
                return (direction, (current_cx, current_cy), edge_point)

        if not multi_frame or self.last_frame is None:
            direction = "none"
            return (direction, (current_cx, current_cy), edge_point)

        last_mask = cv2.inRange(self.last_frame, lower_hsv, upper_hsv)
        last_contours, _ = cv2.findContours(
            last_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not last_contours:
            return ("none", (current_cx, current_cy), (current_cx, current_cy))

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
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down" if dy > 0 else "up"

        if direction == "right":
            edge_point = tuple(contour_points[np.argmax(contour_points[:, 0])])
        elif direction == "left":
            edge_point = tuple(contour_points[np.argmin(contour_points[:, 0])])
        elif direction == "down":
            edge_point = tuple(contour_points[np.argmax(contour_points[:, 1])])
        elif direction == "up":
            edge_point = tuple(contour_points[np.argmin(contour_points[:, 1])])
        else:
            edge_point = (current_cx, current_cy)

        return (direction, (current_cx, current_cy), edge_point)

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