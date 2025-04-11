import yaml
import cv2
from pathlib import Path
from .template import Template

class TemplateManager:
    def __init__(self, template_dir):
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(exist_ok=True)
    
    def create_template(self, template):
        """创建新模板"""
        return template.save(self.template_dir)
    
    def delete_template(self, frame_id, template_id):
        """删除模板"""
        template_path = self.template_dir / frame_id / template_id
        if template_path.exists():
            import shutil
            shutil.rmtree(template_path)
            return True
        return False
    
    def get_template(self, frame_id, template_id):
        """获取模板配置"""
        template_path = self.template_dir / frame_id / template_id
        config_file = template_path / "config.yml"
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return Template.from_dict(config)
        return None
    
    def get_all_templates(self):
        """列出所有模板"""
        templates = []
        for frame_dir in self.template_dir.iterdir():
            if frame_dir.is_dir():
                for template_dir in frame_dir.iterdir():
                    if template_dir.is_dir():
                        config_file = template_dir / "config.yml"
                        if config_file.exists():
                            with open(config_file, 'r', encoding='utf-8') as f:
                                config = yaml.safe_load(f)
                                templates.append(Template.from_dict(config))
        return templates
        
    def save_template(self, frame_id, template_id, template_name, hsv_upper, hsv_lower, crop_area, template_image):
        """保存模板"""
        template_path = self.template_dir / frame_id / template_id
        template_path.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        config = {
            'frame_id': frame_id,
            'template_id': template_id,
            'template_name': template_name,
            'hsv_upper': hsv_upper,
            'hsv_lower': hsv_lower,
            'crop_area': crop_area
        }
        
        with open(template_path / "config.yml", 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
            
        # 保存模板图片
        if template_image is not None:
            cv2.imwrite(str(template_path / "template.png"), template_image)
            
        return True