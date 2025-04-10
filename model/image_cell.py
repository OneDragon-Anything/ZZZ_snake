import cv2

class ImageCell:
    """
    格子类，包含格子中心坐标、边界坐标和HSV色调中的H值数组
    """
    def __init__(self, center=None, bounds=None, image=None):
        """
        初始化格子
        :param center: 中心坐标 (x,y)
        :param bounds: 边界坐标 (x1,y1,x2,y2)
        :param image: 格子图像数据
        """
        self.center = center
        self.bounds = bounds
        self.h_values = None
        self.color_dict = None
        self.center_color = None
        
        if image is not None:
            self.get_center_color(image)
    
    def get_center_color(self, image):
        """
        获取格子中心点的HSV颜色值
        :param image: HSV格式的格子图像数据
        :return: 中心点HSV颜色值
        """
        if image is None or self.center is None:
            return None
            
        cx, cy = self.center
        self.center_color = image[cy, cx].copy()
        return self.center_color
        
    def get_color_dict(self, image=None, border=3):
        """
        获取颜色字典，如果已有则直接返回，否则调用set_image生成
        :param image: HSV格式的格子图像数据
        :param border: 边界像素数，默认为3
        :return: 颜色字典，键为H值，值为出现次数
        """
        if self.color_dict is not None:
            return self.color_dict
            
        if image is None:
            return None
            
        return self.set_image(image, border)
        
    def set_image(self, image, border=3):
        """
        设置格子图像并返回颜色字典
        :param image: HSV格式的格子图像数据
        :param border: 边界像素数，默认为3
        :return: 颜色字典，键为H值，值为出现次数
        """
        if image is None or self.bounds is None:
            return None
            
        x1, y1, x2, y2 = self.bounds
        cropped = image[y1+border:y2-border, x1+border:x2-border]
        h_values = cropped[:,:,0].flatten()
        
        color_dict = {}
        for h in h_values:
            color_dict[h] = color_dict.get(h, 0) + 1
            
        self.h_values = h_values
        self.color_dict = color_dict
        return self.color_dict


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication, QLabel
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPixmap
    
    class DragDropLabel(QLabel):
        def __init__(self):
            super().__init__()
            self.setAcceptDrops(True)
            self.setText("拖入图片文件")
            self.setAlignment(Qt.AlignCenter)
            self.setStyleSheet("""
                QLabel {
                    border: 2px dashed #aaa;
                    padding: 20px;
                }
            """)
        
        def dragEnterEvent(self, event):
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
        
        def dropEvent(self, event):
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    # 记录开始时间
                    import time
                    read_start = time.time()
                    
                    # 读取图像并转换为HSV
                    image = cv2.imread(file_path)
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    
                    # 记录读取结束时间
                    read_end = time.time()
                    
                    # 获取图像尺寸
                    height, width = image.shape[:2]
                    
                    # 创建Cell对象，将整个图像视为一个格子
                    cell = ImageCell(
                        center=(width//2, height//2),
                        bounds=(0, 0, width, height),
                        image=hsv
                    )
                    
                    # 记录分析结束时间
                    analyze_end = time.time()
                    
                    # 获取颜色字典
                    color_dict = cell.set_image(hsv)
                    
                    # 显示处理结果
                    self.setPixmap(QPixmap(file_path).scaled(400, 400, Qt.KeepAspectRatio))
                    print(f"已处理图像: {file_path}")
                    print(f"图像尺寸: {width}x{height}")
                    print(f"Cell中心点: {cell.center}")
                    print(f"中心点HSV值: {cell.center_color}")
                    print(f"裁剪区域: {cell.bounds[0]+3},{cell.bounds[1]+3} - {cell.bounds[2]-3},{cell.bounds[3]-3}")
                    print(f"H值数组长度: {len(cell.h_values) if cell.h_values is not None else 0}")
                    print(f"读取用时: {(read_end - read_start)*1000:.2f}ms")
                    print(f"分析用时: {(analyze_end - read_end)*1000:.2f}ms")
                    print(f"颜色字典: {color_dict}")
                    break
    
    app = QApplication([])
    window = DragDropLabel()
    window.resize(400, 400)
    window.show()
    app.exec_()