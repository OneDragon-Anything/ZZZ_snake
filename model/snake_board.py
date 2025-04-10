import cv2
from .image_cell import ImageCell

class BoardCell(ImageCell):
    """
    棋盘格子类，继承自ImageCell，添加格子类型属性
    """
    def __init__(self, center=None, bounds=None, image=None, cell_type=None, row=None, col=None):
        super().__init__(center, bounds, image)
        self.cell_type = cell_type  # 格子类型属性
        self.row = row  # 格子在棋盘中的行位置
        self.col = col  # 格子在棋盘中的列位置

class Board:
    """
    棋盘类，用于管理游戏棋盘状态和坐标转换
    """
    def __init__(self, rows=25, cols=29, image=None, image_format='HSV'):
        """
        初始化棋盘
        :param rows: 棋盘行数
        :param cols: 棋盘列数
        :param image: 输入图像，默认为None
        :param image_format: 图像格式，可选'BGR'(默认)、'RGB'或'HSV'
        """
        self.rows = rows
        self.cols = cols
        self.rgb_image = None
        self.hsv_image = None
        self.bgr_image = None
        self.cells = None         # 存储BoardCell对象的二维数组
        self.special_cells = {}
        self.head_position = None  # 蛇头精准坐标 (x,y)
        self.head_direction = None  # 蛇头运动方向 ('up','down','left','right')
        
        if image is not None:
            self.set_hsv_image(image, image_format)
        
    def set_hsv_image(self, image, image_format='HSV'):
        """设置原始棋盘图像并预计算所有格子坐标
        :param image: 输入图像
        :param image_format: 图像格式，可选'BGR'(默认)、'RGB'或'HSV'
        """
        # 根据输入格式创建图像副本
        if image_format == 'BGR':
            self.bgr_image = image.copy()
            self.rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif image_format == 'RGB':
            self.rgb_image = image.copy()
            self.bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif image_format == 'HSV':
            self.hsv_image = image.copy()
            self.bgr_image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
            self.rgb_image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        # 初始化所有BoardCell对象并设置中心颜色
        if image is not None:
            h, w = image.shape[:2]
            self.cells = [[BoardCell(self.get_cell_center(r, c), self.get_cell_bounds(r, c), self.hsv_image, None, r, c) 
                         for c in range(self.cols)] for r in range(self.rows)]
            

        
    def set_drawn_image(self, image):
        """设置绘制后的棋盘图像"""
        self.drawn_image = image
        
    def get_cell_center(self, row, col):
        """
        获取指定格子的中心坐标
        :param row: 行索引
        :param col: 列索引
        :return: (x, y) 中心坐标
        """
        if self.hsv_image is None:
            return None
            
        h, w = self.hsv_image.shape[:2]
        cell_h = h / self.rows
        cell_w = w / self.cols
        
        center_x = int(col * cell_w + cell_w / 2)
        center_y = int(row * cell_h + cell_h / 2)
        
        return (center_x, center_y)
    
    def get_cell_bounds(self, row, col):
        """
        获取指定格子的边界坐标
        :param row: 行索引
        :param col: 列索引
        :return: (x1, y1, x2, y2) 边界坐标
        """
        if self.hsv_image is None:
            return None
            
        h, w = self.hsv_image.shape[:2]
        cell_h = h / self.rows
        cell_w = w / self.cols
        
        x1 = int(col * cell_w)
        y1 = int(row * cell_h)
        x2 = int((col + 1) * cell_w)
        y2 = int((row + 1) * cell_h)
        
        return (x1, y1, x2, y2)
        
    def get_cell_by_position(self, x, y):
        """
        根据坐标获取对应的格子对象
        :param x: x坐标
        :param y: y坐标
        :return: BoardCell对象或None(当坐标无效时)
        """
        if self.cells is None or self.hsv_image is None:
            return None
            
        h, w = self.hsv_image.shape[:2]
        cell_h = h / self.rows
        cell_w = w / self.cols
        
        # 计算所在行列
        row = int(y // cell_h)
        col = int(x // cell_w)
        
        # 检查边界
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.cells[row][col]
        return None