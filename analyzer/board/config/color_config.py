# 颜色配置文件
# 存储棋盘分析器使用的HSV颜色常量

class ColorConfig:
    """存储棋盘分析中使用的HSV颜色常量"""
    
    # 默认HSV颜色配置
    DEFAULT_COLORS = {
        "own_head": [1, 213, 255],
        "own_body": [4, 213, 255],
        "own_body_tail": [15, 255, 255],
        "enemy_head": [127, 185, 255],
        "enemy_body_tail": [130, 134, 255],
        "enemy2_head": [150, 250, 255],
        "enemy2_body_tail": [134, 134, 255],
        "enemy3_head": [17, 255, 255],
        "enemy3_body_tail": [30, 255, 255],
        "enemy4_head": [0, 255, 255],
        "enemy4_body": [176, 242, 255],
        "enemy4_body_tail": [157, 152, 255],
        "grid_light": [88, 80, 236],
        "grid_dark": [97, 251, 203],
        "game_over": [109, 191, 88],
    }
    
    @classmethod
    def get_color(cls, color_name):
        """获取指定名称的颜色值"""
        return cls.DEFAULT_COLORS.get(color_name, [0, 0, 0])
    
    @classmethod
    def get_h_value(cls, color_name):
        """获取指定名称颜色的H值"""
        return cls.DEFAULT_COLORS.get(color_name, [0, 0, 0])[0]