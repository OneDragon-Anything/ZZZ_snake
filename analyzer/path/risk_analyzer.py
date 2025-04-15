import numpy as np

class RiskAnalyzer:
    """风险区域分析器，负责计算和管理网格中的风险区域"""
    
    # 风险系数常量
    HIGH_RISK = 81      # 高风险：敌方蛇头、地图边缘等
    MEDIUM_RISK = 9     # 中风险：敌方蛇身、地图外边缘等 
    LOW_RISK = 1        # 低风险：自己蛇身等
    
    def __init__(self, grid_height, grid_width):
        """初始化风险分析器
        Args:
            grid_height: 网格高度
            grid_width: 网格宽度
        """
        self.grid_height = grid_height
        self.grid_width = grid_width
        # 使用NumPy数组存储风险分数，包含边界外一格的缓冲区
        self.risk_array = np.zeros((self.grid_height + 2, self.grid_width + 2), dtype=np.float32)
        # 定义方向向量数组，用于邻居计算
        self.directions = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)], dtype=np.int32)
        # 兼容性字典
        self.risk_scores = {}

    def update_risk_areas(self, board):
        """更新风险区域
        
        Args:
            board: 棋盘对象
        """
        # 重置风险数组
        self.risk_array.fill(0)
        
        # 为了兼容现有代码，保留字典接口
        self.risk_scores = {}
        
        # === 使用NumPy向量化操作标记地图边缘为高风险 ===
        # 上下边缘
        self.risk_array[1, 1:self.grid_width+1] += self.HIGH_RISK  # 上边缘
        self.risk_array[self.grid_height, 1:self.grid_width+1] += self.HIGH_RISK  # 下边缘
        # 左右边缘
        self.risk_array[1:self.grid_height+1, 1] += self.HIGH_RISK  # 左边缘
        self.risk_array[1:self.grid_height+1, self.grid_width] += self.HIGH_RISK  # 右边缘
        
        # === 标记地图边缘外一格为中风险 ===
        # 外部边缘
        self.risk_array[0, :] += self.MEDIUM_RISK  # 上外边缘
        self.risk_array[self.grid_height+1, :] += self.MEDIUM_RISK  # 下外边缘
        self.risk_array[:, 0] += self.MEDIUM_RISK  # 左外边缘
        self.risk_array[:, self.grid_width+1] += self.MEDIUM_RISK  # 右外边缘
        
        # 处理特殊单元格
        self._process_special_cells(board)
        
        # 将NumPy数组的值同步到字典中以保持兼容性
        self._sync_to_dict()
    
    def _process_special_cells(self, board):
        """处理特殊单元格的风险值"""
        
        def process_cells(cell_type, risk_value):
            if cell_type in board.special_cells:
                cells = board.special_cells[cell_type]
                if not cells:
                    return
                    
                # 提取所有单元格的坐标
                coords = np.array([(cell.col, cell.row) for cell in cells])
                if len(coords) == 0:
                    return
                    
                # 为每个坐标生成4个邻居坐标
                for dx, dy in self.directions:
                    # 计算邻居坐标
                    neighbors = coords + np.array([dx, dy])
                    
                    # 过滤有效的邻居坐标
                    valid_mask = (neighbors[:, 0] >= 0) & (neighbors[:, 0] < self.grid_width) & \
                                (neighbors[:, 1] >= 0) & (neighbors[:, 1] < self.grid_height)
                    valid_neighbors = neighbors[valid_mask]
                    
                    # 更新风险值
                    for nx, ny in valid_neighbors:
                        self.risk_array[ny+1, nx+1] += risk_value
        
        # 处理高风险区域 - 敌方蛇头、地雷等周围
        for key in ["enemy_head", "mine", "unknown"]:
            process_cells(key, self.HIGH_RISK)
            
        # 处理中风险区域 - 敌方蛇身周围
        process_cells("enemy_body", self.MEDIUM_RISK)
        
        # 处理低风险区域 - 自己蛇身周围
        for key in ["own_body", "greed_speed"]:
            process_cells(key, self.LOW_RISK)
    
    def _sync_to_dict(self):
        """将风险数组同步到字典中以保持兼容性"""
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                risk_value = self.risk_array[y+1, x+1]
                if risk_value > 0:
                    self.risk_scores[(x, y)] = risk_value
    
    def add_risk_score(self, pos, score):
        """累加风险分数 - 同时更新数组和字典
        
        Args:
            pos: 位置坐标 (x, y)
            score: 风险分数
        """
        x, y = pos
        # 检查坐标是否在扩展的风险数组范围内
        if 0 <= x+1 < self.grid_width+2 and 0 <= y+1 < self.grid_height+2:
            self.risk_array[y+1, x+1] += score
            
        # 更新字典（仅用于兼容现有代码）
        if pos in self.risk_scores:
            self.risk_scores[pos] += score
        else:
            self.risk_scores[pos] = score
    
    def get_risk_score(self, x, y):
        """获取指定位置的风险分数
        
        Args:
            x, y: 坐标
            
        Returns:
            风险分数
        """
        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
            return self.risk_array[y+1, x+1]
        return 0