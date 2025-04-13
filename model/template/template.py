import yaml
import cv2
from pathlib import Path


class Template:
    def __init__(
        self,
        frame_id="",
        template_id="",
        template_name="",
        hsv_color="",
        hsv_diff="",
        crop_area="",
        image_path="",
        area_diff=0.3,
        shape_type="circle",
        shape_diff=0.3,
    ):
        self.frame_id = frame_id
        self.template_id = template_id
        self.template_name = template_name
        self.hsv_color = hsv_color
        self.hsv_diff = hsv_diff
        self.crop_area = crop_area
        self.image_path = image_path
        self.cropped_image = None
        self.area_diff = area_diff  # 新增：面积差异
        self.shape_type = shape_type  # 新增：形状类型
        self.shape_diff = shape_diff  # 新增：形状差异

    def to_dict(self):
        return {
            "frame_id": self.frame_id,
            "template_id": self.template_id,
            "template_name": self.template_name,
            "hsv_color": self.hsv_color,
            "hsv_diff": self.hsv_diff,
            "crop_area": self.crop_area,
            "image_path": self.image_path,
            "area_diff": self.area_diff,  # 新增
            "shape_type": self.shape_type,  # 新增
            "shape_diff": self.shape_diff,  # 新增
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            frame_id=data.get("frame_id", ""),
            template_id=data.get("template_id", ""),
            template_name=data.get("template_name", ""),
            hsv_color=data.get("hsv_color", ""),
            hsv_diff=data.get("hsv_diff", ""),
            crop_area=data.get("crop_area", ""),
            image_path=data.get("image_path", ""),
            area_diff=data.get("area_diff", 0.1),  # 新增
            shape_type=data.get("shape_type", "circle"),  # 新增
            shape_diff=data.get("shape_diff", 0.1),  # 新增
        )

    def save(self, template_dir, template_image=None):
        """保存模板到指定目录
        Args:
            template_dir: 模板目录
            template_image: 要保存的模板图片(已处理好的)
        """
        template_path = Path(template_dir) / self.frame_id / self.template_id
        template_path.mkdir(parents=True, exist_ok=True)

        # 保存配置
        with open(template_path / "config.yml", "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True)

        # 保存模板图片
        if template_image is not None:
            cv2.imwrite(str(template_path / "template.png"), template_image)
