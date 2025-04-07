import cv2
import numpy as np
from PIL import Image

class MapDrawer:
    def __init__(self):
        pass

    def draw_map(self, board_state, path, screen_cv, cell_size=None):
        """
        绘制游戏地图，包括网格线、路径和特殊格子
        
        Args:
            board_state: 棋盘状态，二维列表
            path: 路径列表，每个元素为 (x, y) 坐标元组
            screen_cv: OpenCV 格式的原始图像
            cell_size: 可选，格子大小元组 (width, height)，如果不提供则自动计算
            
        Returns:
            处理后的 OpenCV 格式图像
        """
        if screen_cv is None:
            return None
            
        # 获取图像尺寸
        h, w = screen_cv.shape[:2]
        
        # 如果没有提供格子大小，根据棋盘状态计算
        if cell_size is None and board_state is not None:
            grid_h = len(board_state)  # 棋盘高度（行数）
            grid_w = len(board_state[0]) if grid_h > 0 else 0  # 棋盘宽度（列数）
            if grid_h > 0 and grid_w > 0:
                cell_h = h / grid_h
                cell_w = w / grid_w
                cell_size = (cell_w, cell_h)
            
        if cell_size is None:
            return screen_cv
            
        cell_w, cell_h = cell_size
        
        # 绘制网格线
        self._draw_grid(screen_cv, cell_size)
        
        # 绘制特殊格子
        if board_state is not None:
            self._draw_special_cells(screen_cv, board_state, cell_size)
        
        # 绘制路径
        if path and len(path) > 1:
            self._draw_path(screen_cv, path, cell_size)
            
        return screen_cv
    
    def _draw_grid(self, screen_cv, cell_size):
        """绘制网格线"""
        h, w = screen_cv.shape[:2]
        cell_w, cell_h = cell_size
        
        # 计算网格数量
        grid_h = int(h / cell_h)
        grid_w = int(w / cell_w)
        
        # 绘制横线
        for i in range(grid_h + 1):
            y = int(i * cell_h)
            cv2.line(screen_cv, (0, y), (w, y), (255, 255, 255), 1)
            
        # 绘制竖线
        for j in range(grid_w + 1):
            x = int(j * cell_w)
            cv2.line(screen_cv, (x, 0), (x, h), (255, 255, 255), 1)
    
    def _draw_special_cells(self, screen_cv, board_state, cell_size):
        """绘制特殊格子"""
        cell_w, cell_h = cell_size
        
        for y, row in enumerate(board_state):
            for x, cell_type in enumerate(row):
                if cell_type in ['speed_boost', 'score_boost', 'own_head', 'own_body', 'enemy_head', 'enemy_body']:
                    # 计算格子中心点坐标
                    center_x = int(x * cell_w + cell_w / 2)
                    center_y = int(y * cell_h + cell_h / 2)
                    
                    # 根据格子类型设置颜色
                    if cell_type == 'speed_boost':
                        color = (0, 255, 0)  # 绿色
                    elif cell_type == 'score_boost':
                        color = (255, 165, 0)  # 橙色
                    elif cell_type == 'own_head':
                        color = (0, 255, 255)  # 黄色
                    elif cell_type == 'own_body':
                        color = (255, 255, 255)  # 白色
                    elif cell_type == 'enemy_head':
                        color = (255, 0, 0)  # 红色
                    elif cell_type == 'enemy_body':
                        color = (128, 0, 0)  # 深红色
                    
                    # 在格子中心绘制圆点
                    cv2.circle(screen_cv, (center_x, center_y), 4, color, -1)
    
    def _draw_path(self, screen_cv, path, cell_size):
        """绘制路径"""
        cell_w, cell_h = cell_size
        
        # 绘制起点（绿色大圆）
        start_x = int(path[0][0] * cell_w + cell_w / 2)
        start_y = int(path[0][1] * cell_h + cell_h / 2)
        cv2.circle(screen_cv, (start_x, start_y), 8, (0, 255, 0), -1)
        
        # 绘制路径线段
        for i in range(len(path)-1):
            start_x = int(path[i][0] * cell_w + cell_w / 2)
            start_y = int(path[i][1] * cell_h + cell_h / 2)
            end_x = int(path[i+1][0] * cell_w + cell_w / 2)
            end_y = int(path[i+1][1] * cell_h + cell_h / 2)
            cv2.arrowedLine(screen_cv, (start_x, start_y), (end_x, end_y), 
                           (255, 0, 0), 2, tipLength=0.3)
        
        # 绘制终点（红色大圆）
        end_x = int(path[-1][0] * cell_w + cell_w / 2)
        end_y = int(path[-1][1] * cell_h + cell_h / 2)
        cv2.circle(screen_cv, (end_x, end_y), 8, (0, 0, 255), -1)