import yaml
import cv2
import numpy as np
from pathlib import Path
from .template import Template

class TemplateManager:
    def __init__(self, template_dir):
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(exist_ok=True)
        self._hsv_cache = {}
        self._use_opencl = cv2.ocl.haveOpenCL()
        if self._use_opencl:
            cv2.ocl.setUseOpenCL(True)
    
    def create_template(self, template):
        return template.save(self.template_dir)
    
    def delete_template(self, frame_id, template_id):
        template_path = self.template_dir / frame_id / template_id
        if template_path.exists():
            import shutil
            shutil.rmtree(template_path)
            return True
        return False
    
    def get_template(self, frame_id, template_id):
        template_path = self.template_dir / frame_id / template_id
        config_file = template_path / "config.yml"
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                image_path = template_path / "template.png"
                if image_path.exists():
                    config['image_path'] = str(image_path)
                return Template.from_dict(config)
        return None
    
    def get_all_templates(self):
        templates = []
        for frame_dir in self.template_dir.iterdir():
            if frame_dir.is_dir():
                for template_dir in frame_dir.iterdir():
                    if template_dir.is_dir():
                        config_file = template_dir / "config.yml"
                        if config_file.exists():
                            with open(config_file, 'r', encoding='utf-8') as f:
                                config = yaml.safe_load(f)
                                template = Template.from_dict(config)
                                image_path = template_dir / "template.png"
                                if image_path.exists():
                                    template.image_path = str(image_path)
                                templates.append(template)
        return templates
        
    def save_template(self, frame_id, template_id, template_name, hsv_color, hsv_diff, crop_area, template_image):
        template_path = self.template_dir / frame_id / template_id
        template_path.mkdir(parents=True, exist_ok=True)
        
        config = {
            'frame_id': frame_id,
            'template_id': template_id,
            'template_name': template_name,
            'hsv_color': hsv_color,
            'hsv_diff': hsv_diff,
            'crop_area': crop_area
        }
        
        with open(template_path / "config.yml", 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
            
        if template_image is not None:
            cv2.imwrite(str(template_path / "template.png"), template_image)
            
        return True

    def _get_hsv_ranges(self, template):
        """获取模板的HSV范围，使用模板对象作为缓存键"""
        if not hasattr(template, '_hsv_ranges_cache'):
            # 从模板对象获取HSV参数
            hsv_color = template.hsv_color
            hsv_diff = template.hsv_diff
            
            # 处理字符串或元组形式的输入
            if isinstance(hsv_color, str):
                h_color, s_color, v_color = map(int, hsv_color.split(','))
            else:
                h_color, s_color, v_color = hsv_color
                
            if isinstance(hsv_diff, str):
                h_diff, s_diff, v_diff = map(int, hsv_diff.split(','))
            else:
                h_diff, s_diff, v_diff = hsv_diff

            # 计算HSV范围
            if h_diff >= 90:
                h_lower, h_upper = 0, 179
            else:
                h_lower = (h_color - h_diff) % 180
                h_upper = (h_color + h_diff) % 180

            s_lower = max(0, s_color - s_diff)
            s_upper = min(255, s_color + s_diff)
            v_lower = max(0, v_color - v_diff)
            v_upper = min(255, v_color + v_diff)

            # 缓存到模板对象
            template._hsv_ranges_cache = {
                'h_lower': h_lower,
                'h_upper': h_upper,
                's_lower': s_lower,
                's_upper': s_upper,
                'v_lower': v_lower,
                'v_upper': v_upper
            }
        
        return template._hsv_ranges_cache

    def _get_hsv_mask(self, hsv_img, hsv_ranges):
        # 预计算边界数组（避免重复创建）
        lower = np.array([hsv_ranges['h_lower'], hsv_ranges['s_lower'], hsv_ranges['v_lower']], dtype=np.uint8)
        upper = np.array([hsv_ranges['h_upper'], hsv_ranges['s_upper'], hsv_ranges['v_upper']], dtype=np.uint8)
        
        if self._use_opencl:
            # OpenCL路径优化
            hsv = cv2.UMat(hsv_img) if not isinstance(hsv_img, cv2.UMat) else hsv_img
            if hsv_ranges['h_lower'] > hsv_ranges['h_upper']:
                mask1 = cv2.inRange(hsv, np.array([0, *lower[1:]]), np.array([hsv_ranges['h_upper'], *upper[1:]]))
                mask2 = cv2.inRange(hsv, np.array([hsv_ranges['h_lower'], *lower[1:]]), np.array([179, *upper[1:]]))
                return cv2.bitwise_or(mask1, mask2).get()
            return cv2.inRange(hsv, lower, upper).get()
        else:
            # 普通路径优化
            if hsv_ranges['h_lower'] > hsv_ranges['h_upper']:
                mask1 = cv2.inRange(hsv_img, np.array([0, *lower[1:]]), np.array([hsv_ranges['h_upper'], *upper[1:]]))
                mask2 = cv2.inRange(hsv_img, np.array([hsv_ranges['h_lower'], *lower[1:]]), np.array([179, *upper[1:]]))
                return cv2.bitwise_or(mask1, mask2)
            return cv2.inRange(hsv_img, lower, upper)

    def get_template_analysis(self, template):
        """获取模板分析结果，如果已缓存则直接返回"""
        if hasattr(template, '_analysis_cache'):
            return template._analysis_cache
            
        # 初始化分析缓存
        template._analysis_cache = {
            'area': 0,
            'shape_metric': 0,
            'contour': None,
            'hsv_img': None,
            'mask': None
        }
        
        if hasattr(template, 'image_path') and template.image_path:
            # 读取并缓存图像
            if hasattr(template, 'cropped_image') and template.cropped_image is not None:
                template_img = template.cropped_image
            else:
                template_img = cv2.imread(template.image_path)
                template.cropped_image = template_img
            
            # 缓存HSV图像
            template._analysis_cache['hsv_img'] = cv2.cvtColor(template_img, cv2.COLOR_BGR2HSV)
            
            # 计算并缓存掩码 (修改后的调用方式)
            hsv_ranges = self._get_hsv_ranges(template)
            template._analysis_cache['mask'] = self._get_hsv_mask(
                template._analysis_cache['hsv_img'], 
                hsv_ranges
            )
            
            # 查找并缓存轮廓
            contours, _ = cv2.findContours(
                template._analysis_cache['mask'],
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                template._analysis_cache['contour'] = max(contours, key=cv2.contourArea)
                template._analysis_cache['area'] = cv2.contourArea(template._analysis_cache['contour'])
                
                # 计算形状指标
                perimeter = cv2.arcLength(template._analysis_cache['contour'], True)
                if perimeter > 0:
                    if template.shape_type == "circle":
                        template._analysis_cache['shape_metric'] = 4 * np.pi * template._analysis_cache['area'] / (perimeter**2)
                    elif template.shape_type == "rectangle":
                        _, (w, h), _ = cv2.minAreaRect(template._analysis_cache['contour'])
                        rect_area = w * h
                        if rect_area > 0:
                            template._analysis_cache['shape_metric'] = template._analysis_cache['area'] / rect_area

        return template._analysis_cache

    def find_objects_by_features(self, template, image=None, hsv_image=None, search_in_crop=False):
        try:
            # 使用get_template_analysis获取模板分析结果
            analysis = self.get_template_analysis(template)
            template_area = analysis['area']
            template_shape_metric = analysis['shape_metric']

            # 处理目标图像
            if hsv_image is not None:
                hsv = hsv_image
            elif image is not None:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            else:
                raise ValueError("必须提供image或hsv_image参数")
            hsv_ranges = self._get_hsv_ranges(template)
            
            mask = self._get_hsv_mask(hsv, hsv_ranges)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            results = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 10:
                    continue
                
                # 轮廓分析计算
                if template.shape_type == "circle":
                    (x,y), _ = cv2.minEnclosingCircle(cnt)
                    center = (int(x), int(y))
                elif template.shape_type == "rectangle":
                    rect = cv2.minAreaRect(cnt)
                    center = (int(rect[0][0]), int(rect[0][1]))
                else:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    else:
                        continue
                
                perimeter = cv2.arcLength(cnt, True)
                
                if template.shape_type == "circle":
                    shape_metric = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
                elif template.shape_type == "rectangle":
                    _, (w, h), _ = cv2.minAreaRect(cnt)
                    rect_area = w * h
                    shape_metric = area / rect_area if rect_area > 0 else 0
                else:
                    shape_metric = (perimeter**2) / area if area > 0 else 0

                area_score = 0.0
                shape_score = 0.0
                
                if template_area > 0:
                    area_ratio = area / template_area
                    area_score = max(0, min(1.0, 1 - abs(1 - area_ratio)))
                    
                    if template_shape_metric > 0:
                        shape_ratio = shape_metric / template_shape_metric
                        shape_score = max(0, min(1.0, 1 - abs(1 - shape_ratio)))
                
                score = area_score * shape_score if template_area > 0 else shape_metric
                
                if score > 0.5:
                    results.append((center[0], center[1], 0, score))
            return results
            
        except Exception as e:
            print(f"对象检测出错: {str(e)}")
            return []

    def filter_image_by_hsv(self, image, hsv_color, hsv_diff):
        """根据HSV值过滤图像并返回二值化结果
        Args:
            image: 输入图像(BGR格式)
            hsv_color: HSV颜色值元组(H,S,V)
            hsv_diff: HSV色差元组(H_diff,S_diff,V_diff)
        Returns:
            二值化掩码图像
        """
        # 转换为HSV颜色空间
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 创建临时模板对象
        temp_template = Template(
            frame_id="temp",
            template_id="temp",
            template_name="temp",
            hsv_color=hsv_color,
            hsv_diff=hsv_diff,
            crop_area=None,
            shape_type="rectangle"  # 任意形状类型
        )
        
        # 获取HSV范围
        hsv_ranges = self._get_hsv_ranges(temp_template)
        
        # 获取HSV掩码
        mask = self._get_hsv_mask(hsv_img, hsv_ranges)
        
        # 应用形态学操作去除噪声
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 将二值化掩码转换为3通道图像以便显示
        result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return result