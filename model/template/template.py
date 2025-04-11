import yaml
from pathlib import Path

class Template:
    def __init__(self, frame_id='', template_id='', name='', hsv_upper='', hsv_lower='', crop_area=''):
        self.frame_id = frame_id
        self.template_id = template_id
        self.name = name
        self.hsv_upper = hsv_upper
        self.hsv_lower = hsv_lower
        self.crop_area = crop_area
        
    def to_dict(self):
        return {
            'frame_id': self.frame_id,
            'template_id': self.template_id,
            'template_name': self.name,
            'hsv_upper': self.hsv_upper,
            'hsv_lower': self.hsv_lower,
            'crop_area': self.crop_area
        }
        
    @classmethod
    def from_dict(cls, data):
        return cls(
            frame_id=data.get('frame_id', ''),
            template_id=data.get('template_id', ''),
            name=data.get('template_name', ''),
            hsv_upper=data.get('hsv_upper', ''),
            hsv_lower=data.get('hsv_lower', ''),
            crop_area=data.get('crop_area', '')
        )
        
    def save(self, template_dir):
        """保存模板到指定目录"""
        template_path = Path(template_dir) / self.frame_id / self.template_id
        template_path.mkdir(parents=True, exist_ok=True)
        
        with open(template_path / "config.yml", 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True)