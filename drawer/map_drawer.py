import cv2
import os
import time

class MapDrawer:
    def __init__(self, logger=None):
        self.logger = logger
        self.pro = False
        self.board = None

    def draw_map(self, board, path=None):
        """
        绘制游戏地图，包括网格线、路径和特殊格子
        
        Args:
            board: Board对象，包含图像和格子信息
            path: 路径列表，每个元素为 (x, y) 坐标元组
            
        Returns:
            处理后的 OpenCV 格式图像
        """
        if board is None or board.rgb_image is None:
            return None
            
        self.board = board
        # 创建绘图画布
        screen_cv = board.bgr_image.copy()
        
        # 绘制网格线
        self._draw_grid(screen_cv)
        
        # 绘制特殊格子
        if board.cells is not None:
            self._draw_special_cells(screen_cv)
        
        # 绘制路径
        if path and len(path) > 1:
            self._draw_path(screen_cv, path)
            
        # 调用绘制HSV色调方法
        if self.pro == True:
            self._draw_cell_hsv(screen_cv)
            
            # 保存调试图片
            debug_dir = os.path.join(os.path.dirname(__file__), '.debug', 'images')
            try:
                os.makedirs(debug_dir, exist_ok=True)
                
                timestamp = int(time.time() * 1000)
                debug_path = os.path.join(debug_dir, f'debug_{timestamp}.png')
                
                success = cv2.imwrite(debug_path, screen_cv)
                if success:
                    self.logger.log(f'调试图片已保存至: {debug_path}')
                else:
                    # 添加更详细的错误信息
                    error_msg = f'无法保存调试图片到: {debug_path}'
                    if not os.path.exists(debug_dir):
                        error_msg += ' (目录不存在)'
                    elif not os.access(debug_dir, os.W_OK):
                        error_msg += ' (目录不可写)'
                    elif not os.path.isfile(debug_path):
                        error_msg += ' (文件创建失败)'
                    self.logger.log(error_msg)
            except Exception as e:
                self.logger.log(f'保存调试图片时发生错误: {str(e)}')

        return screen_cv
    
    def _draw_grid(self, screen_cv):
        """绘制网格线"""
        h, w = screen_cv.shape[:2]
        
        # 使用Board的行列数
        for i in range(1, self.board.rows):
            bounds = self.board.get_cell_bounds(i, 0)
            y = bounds[1]  # y1坐标
            cv2.line(screen_cv, (0, y), (w, y), (255, 255, 255), 1)
            
        for j in range(1, self.board.cols):
            bounds = self.board.get_cell_bounds(0, j)
            x = bounds[0]  # x1坐标
            cv2.line(screen_cv, (x, 0), (x, h), (255, 255, 255), 1)
    
    def _draw_special_cells(self, screen_cv):
        """绘制所有非空格子并自动分配颜色"""
        color_map = {
            'speed_boost': (55, 55, 55),    # 深灰色
            'score_boost': (255, 165, 0),   # 橙色
            'own_head': (0, 255, 0),       # 绿色
            'own_body': (255, 255, 255),   # 白色
            'own_tail': (255, 50, 50),     # 蓝色
            'enemy_head': (0, 255, 255),   # 黄色
            'enemy_body': (128, 0, 0),     # 深红色
            'enemy_tail': (128, 128, 0),   # 橄榄色
            'mine': (255, 0, 255),         # 紫色
            'unknow': (0, 0, 0),           # 黑色
            # 默认颜色
            'default': (128, 128, 128)     # 灰色
        }

        for y, row in enumerate(self.board.cells):
            for x, cell in enumerate(row):
                if cell.cell_type == 'empty':
                    continue
                    
                center = self.board.get_cell_center(y, x)
                if center is None:
                    continue
                    
                center_x, center_y = center
                color = color_map.get(cell.cell_type, color_map['default'])
                
                # 在格子中心绘制X标记
                size = 10
                cv2.line(screen_cv, (center_x - size, center_y - size), 
                        (center_x + size, center_y + size), color, 3)
                cv2.line(screen_cv, (center_x - size, center_y + size), 
                        (center_x + size, center_y - size), color, 3)
    
    def _draw_cell_hsv(self, screen_cv):
        """绘制格子色调并显示HSV数值(带描边效果)"""
        font = cv2.FONT_HERSHEY_DUPLEX
        try:
            font = cv2.FONT_HERSHEY_COMPLEX  # 尝试使用更接近微软雅黑的字体
        except:
            pass
        font_scale = 0.5
        font_thickness = 2
        
        for y, row in enumerate(self.board.cells):
            for x, cell in enumerate(row):
                # 只绘制非empty类型的格子
                if cell.cell_type == 'empty':
                    continue
                # 获取格子中心点的HSV颜色值
                hsv_color = cell.get_center_color(self.board.hsv_image)
                if hsv_color is None:
                    continue
                    
                # 获取H通道的值
                h_text = f"{hsv_color[0]}"
                
                # 获取文字大小和位置
                center_x, center_y = cell.center
                text_size = cv2.getTextSize(h_text, font, font_scale, font_thickness)[0]
                text_x = center_x - text_size[0] // 2
                text_y = center_y + text_size[1] // 2
                
                # 绘制黑色描边(上下左右各偏移1像素)
                offsets = [(-1,-1), (-1,1), (1,-1), (1,1), (0,-1), (0,1), (-1,0), (1,0)]
                for dx, dy in offsets:
                    cv2.putText(screen_cv, h_text, 
                               (text_x + dx, text_y + dy), 
                               font, font_scale, 
                               (0, 0, 0),  # 黑色描边
                               font_thickness)
                
                # 绘制白色文字
                cv2.putText(screen_cv, h_text, 
                           (text_x, text_y), 
                           font, font_scale, 
                           (255, 255, 255),  # 白色文字
                           font_thickness)
    
    def _draw_path(self, screen_cv, path):
        """绘制路径"""
        if not path:
            return
            
        # 绘制起点（绿色大圆）
        start = self.board.get_cell_center(path[0][1], path[0][0])
        if start:
            start_x, start_y = start
            cv2.circle(screen_cv, (start_x, start_y), 8, (0, 255, 0), -1)
        
        # 绘制路径线段
        for i in range(len(path)-1):
            start = self.board.get_cell_center(path[i][1], path[i][0])
            end = self.board.get_cell_center(path[i+1][1], path[i+1][0])
            if start and end:
                start_x, start_y = start
                end_x, end_y = end
            cv2.arrowedLine(screen_cv, (start_x, start_y), (end_x, end_y), 
                           (255, 0, 0), 2, tipLength=0.3)
        
        # 绘制终点（红色大圆）
        end = self.board.get_cell_center(path[-1][1], path[-1][0])
        if end:
            end_x, end_y = end
            cv2.circle(screen_cv, (end_x, end_y), 8, (0, 0, 255), -1)