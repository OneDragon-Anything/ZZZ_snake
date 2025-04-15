import cv2
import numpy as np
from pathlib import Path
from model.template.template import Template
from model.template.template_manage import TemplateManager


class TemplateAnalyzer:
    """负责模板匹配分析的类，从BoardAnalyzer中分离出来"""

    def __init__(self, template_dir=None):
        # 初始化模板管理器和模板缓存
        if template_dir is None:
            template_dir = Path(__file__).parent.parent.parent / "templates"
        self.template_manager = TemplateManager(template_dir)
        self._cached_templates = {}
        
        # 预加载常用模板
        self._preload_common_templates()
    
    def _preload_common_templates(self):
        """预加载常用模板，提高性能"""
        common_templates = ['diamond', 'greed_speed', 'mine', 'super_star', 'yellow_crystal']
        for template_name in common_templates:
            template = self.template_manager.get_template('snake', template_name)
            if template:
                self._cached_templates[template_name] = template
    
    def analyze_by_templates(self, board):
        """使用模板管理器分析棋盘并更新格子类型
        参数:
            board: 棋盘对象
        返回:
            匹配到的特殊格子字典 {类型: [(x, y, 角度, 分数), ...]}
        """
        if not board or not hasattr(board, 'hsv_image'):
            return {}

        # 使用初始化时缓存的模板
        all_matches = {}
        for template_name, template in self._cached_templates.items():
            category = template_name
            if category in ["diamond", "yellow_crystal"]:
                category = "score_boost"
            
            matches = self.template_manager.find_objects_by_features(
                template, hsv_image=board.hsv_image
            )
            if matches:
                if category not in all_matches:
                    all_matches[category] = []
                all_matches[category].extend(matches)

        return all_matches
    
    def get_cells_to_update(self, board, matches, cell_type):
        """根据匹配结果获取需要更新的格子
        参数:
            board: 棋盘对象
            matches: 匹配结果列表 [(x, y, 角度, 分数), ...]
            cell_type: 格子类型
        返回:
            需要更新的格子列表
        """
        is_special = cell_type in ["mine", "super_star"]
        cells_to_update = []

        for x, y, angle, score in matches:
            if is_special:
                # 对于特殊物品，检查周围多个点
                positions = [
                    (x + 5, y + 5), (x - 5, y - 5),
                    (x - 5, y + 5), (x + 5, y - 5)
                ]
                cells = [board.get_cell_by_position(px, py) for px, py in positions]
                cells_to_update.extend([c for c in cells if c])
            else:
                cell = board.get_cell_by_position(x, y)
                if cell:
                    cells_to_update.append(cell)

        return cells_to_update