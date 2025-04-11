import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QLabel
from qfluentwidgets import ComboBox, Slider, LineEdit
from qfluentwidgets import (
    BodyLabel, CaptionLabel, TextBrowser, PrimaryPushButton, InfoBadge,
    CardWidget, SimpleCardWidget
)
import os
import yaml
from pathlib import Path
from log.log import SnakeLogger

class TemplateCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('TemplateCard')
        
        # 主布局 - 垂直分为顶部状态栏和下方内容
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 顶部状态栏
        self.status_bar = CardWidget()
        self.status_bar.setObjectName('StatusBarCard')
        status_layout = QHBoxLayout(self.status_bar)
        status_layout.setContentsMargins(10, 5, 10, 5)
        self.status_label = BodyLabel("状态: 准备就绪")
        status_layout.addWidget(self.status_label)
        main_layout.addWidget(self.status_bar)
        
        # 内容区域 - 水平分为左右两部分
        content_frame = QFrame()
        content_layout = QHBoxLayout(content_frame)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(20)
        
        # 左侧工具栏卡片
        left_card = CardWidget()
        left_card.setFixedWidth(300)
        left_layout = QVBoxLayout(left_card)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(15, 15, 15, 15)
        left_layout.setAlignment(Qt.AlignTop)
        
        # 1. 模板选择下拉框
        template_combo_layout = QHBoxLayout()
        self.template_combo = ComboBox()
        self.new_button = PrimaryPushButton('新建')
        template_combo_layout.addWidget(self.template_combo)
        template_combo_layout.addWidget(self.new_button)
        left_layout.addLayout(template_combo_layout)
        
        # 2. 按钮区域
        button_layout = QHBoxLayout()
        self.open_button = PrimaryPushButton('读取画面')
        self.save_button = PrimaryPushButton('保存')
        self.delete_button = PrimaryPushButton('删除')
        button_layout.addWidget(self.open_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.delete_button)
        left_layout.addLayout(button_layout)
        
        # 3. 缩放滚动条
        zoom_layout = QHBoxLayout()
        self.zoom_scroll = Slider(Qt.Horizontal)
        self.zoom_scroll.setRange(10, 200)
        self.zoom_scroll.setValue(100)
        zoom_layout.addWidget(self.zoom_scroll)
        left_layout.addLayout(zoom_layout)
        left_layout.addSpacing(15)
        
        # 3-7. 各种ID和HSV设置
        self.frame_id_edit = self._create_input_row('画面ID:', left_layout)
        self.template_id_edit = self._create_input_row('模板ID:', left_layout)
        self.template_name_edit = self._create_input_row('模板名称:', left_layout)
        left_layout.addSpacing(15)
        # HSV上限设置
        hsv_upper_layout = QHBoxLayout()
        hsv_upper_layout.addWidget(BodyLabel('HSV上限:'))
        self.h_upper_edit = LineEdit()
        self.h_upper_edit.setPlaceholderText('H(0-179)')
        self.s_upper_edit = LineEdit()
        self.s_upper_edit.setPlaceholderText('S(0-255)')
        self.v_upper_edit = LineEdit()
        self.v_upper_edit.setPlaceholderText('V(0-255)')
        hsv_upper_layout.addWidget(self.h_upper_edit)
        hsv_upper_layout.addWidget(self.s_upper_edit)
        hsv_upper_layout.addWidget(self.v_upper_edit)
        left_layout.addLayout(hsv_upper_layout)
        
        # HSV下限设置
        hsv_lower_layout = QHBoxLayout()
        hsv_lower_layout.addWidget(BodyLabel('HSV下限:'))
        self.h_lower_edit = LineEdit()
        self.h_lower_edit.setPlaceholderText('H(0-179)')
        self.s_lower_edit = LineEdit()
        self.s_lower_edit.setPlaceholderText('S(0-255)')
        self.v_lower_edit = LineEdit()
        self.v_lower_edit.setPlaceholderText('V(0-255)')
        hsv_lower_layout.addWidget(self.h_lower_edit)
        hsv_lower_layout.addWidget(self.s_lower_edit)
        hsv_lower_layout.addWidget(self.v_lower_edit)
        left_layout.addLayout(hsv_lower_layout)
        
        # 8. HSV应用按钮
        hsv_apply_layout = QHBoxLayout()
        self.apply_crop_button = PrimaryPushButton('分割裁剪区域')
        self.apply_original_button = PrimaryPushButton('分割完整图像')
        self.apply_crop_button.clicked.connect(self._apply_hsv_crop)
        self.apply_original_button.clicked.connect(self._apply_hsv_original)
        hsv_apply_layout.addWidget(self.apply_crop_button)
        hsv_apply_layout.addWidget(self.apply_original_button)
        left_layout.addLayout(hsv_apply_layout)
        
        left_layout.addSpacing(15)
        # 9. 裁剪按钮和文本框
        crop_layout = QHBoxLayout()
        self.crop_button = PrimaryPushButton('裁剪')
        self.crop_button.clicked.connect(self._toggle_crop_mode)
        self.position_button = PrimaryPushButton('定位')
        self.crop_text = LineEdit()
        self.is_crop_mode = False
        crop_layout.addWidget(self.crop_button)
        crop_layout.addWidget(self.position_button)
        crop_layout.addWidget(self.crop_text)
        left_layout.addLayout(crop_layout)
        
        # 9. 裁剪预览框
        self.crop_preview_label = QLabel()
        self.crop_preview_label.setFixedSize(200, 200)
        self.crop_preview_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.crop_preview_label)
        
        # 右侧图像显示区域卡片
        right_card = SimpleCardWidget()
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(10, 10, 10, 10)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.mousePressEvent = self._on_image_clicked
        right_layout.addWidget(self.image_label)
        
        # 将左右卡片加入内容布局
        content_layout.addWidget(left_card)
        content_layout.addWidget(right_card, stretch=1)
        
        # 将内容区域加入主布局
        main_layout.addWidget(content_frame, stretch=1)
        
        # 初始化logger
        self.logger = SnakeLogger(None)
        
        # 初始化模板目录
        self.template_dir = Path(__file__).parent.parent.parent / "templates"
        self.template_dir.mkdir(exist_ok=True)
        
        # 连接按钮信号
        self.new_button.clicked.connect(self._on_new_template)
        self.open_button.clicked.connect(self._on_read_frame)
        self.save_button.clicked.connect(self._on_save_template)
        self.delete_button.clicked.connect(self._on_delete_template)
        
        # 初始禁用按钮
        self._update_button_states(False)
    
    def _create_input_row(self, label_text, parent_layout):
        """创建带标签的输入行"""
        row_layout = QHBoxLayout()
        label = BodyLabel(label_text)
        edit = LineEdit()
        row_layout.addWidget(label)
        row_layout.addWidget(edit)
        parent_layout.addLayout(row_layout)
        return edit
        
    def _toggle_crop_mode(self):
        """切换裁剪/恢复模式"""
        self.is_crop_mode = not self.is_crop_mode
        if self.is_crop_mode:
            self.crop_button.setText('请框选')
            self.status_label.setText("状态: 裁剪模式 - 请点击图像选择起点")
            self.start_pos = None
            self.end_pos = None
        else:
            self.crop_button.setText('裁剪')
            if self.start_pos and self.end_pos:
                x1, y1 = self.start_pos
                x2, y2 = self.end_pos
                self.crop_text.setText(f"{x1},{y1},{x2},{y2}")
            self.status_label.setText("状态: 裁剪完成")
            
    def _load_image_file(self, image_path):
        """独立图片加载方法"""
        try:
            self._load_image(image_path)
            self.status_label.setText("状态: 图片加载成功")
            # 启用保存按钮
            self._update_button_states(True)
        except Exception as e:
            self.status_label.setText(f"状态: 错误 - 加载图片失败: {str(e)}")
            self.logger.error(f"加载图片失败: {str(e)}")
            
    def _on_read_frame(self):
        """读取画面按钮点击事件"""
        from PyQt5.QtWidgets import QFileDialog
        
        # 打开文件对话框选择PNG图片
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择图片文件", 
            "", 
            "PNG图片 (*.png);;所有文件 (*)"
        )
        
        if file_path:
            try:
                self._load_image_file(file_path)
                self.status_label.setText(f"状态: 已加载图片 {file_path}")
            except Exception as e:
                self.status_label.setText(f"状态: 错误 - 加载图片失败: {str(e)}")
                self.logger.error(f"加载图片失败: {str(e)}")
            
    def _on_save_template(self):
        """保存模板按钮点击事件"""
        # 获取输入数据
        frame_id = self.frame_id_edit.text().strip()
        template_id = self.template_id_edit.text().strip()
        template_name = self.template_name_edit.text().strip()
        
        # 验证HSV输入
        try:
            h_upper = int(self.h_upper_edit.text().strip())
            s_upper = int(self.s_upper_edit.text().strip())
            v_upper = int(self.v_upper_edit.text().strip())
            h_lower = int(self.h_lower_edit.text().strip())
            s_lower = int(self.s_lower_edit.text().strip())
            v_lower = int(self.v_lower_edit.text().strip())
            
            if not (0 <= h_upper <= 179 and 0 <= h_lower <= 179):
                raise ValueError("H值必须在0-179之间")
            if not (0 <= s_upper <= 255 and 0 <= s_lower <= 255 and 
                    0 <= v_upper <= 255 and 0 <= v_lower <= 255):
                raise ValueError("S和V值必须在0-255之间")
                
            hsv_upper = f"{h_upper},{s_upper},{v_upper}"
            hsv_lower = f"{h_lower},{s_lower},{v_lower}"
        except ValueError as e:
            self.status_label.setText(f"状态: 错误 - {str(e)}")
            return
            
        crop_area = self.crop_text.text().strip()
        
        # 验证必填字段
        if not all([frame_id, template_id, template_name]):
            self.status_label.setText("状态: 错误 - 画面ID、模板ID和名称不能为空")
            return
            
        # 创建模板目录
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
        
        try:
            with open(template_path / "config.yml", 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True)
                
            # 保存模板图片
            if hasattr(self, 'current_image') and self.current_image is not None:
                cv2.imwrite(str(template_path / "template.png"), self.current_image)
                
            # 更新模板下拉列表
            self._update_template_combo()
            
            self.status_label.setText(f"状态: 已保存模板 {template_name}")
        except Exception as e:
            self.status_label.setText(f"状态: 错误 - 保存模板失败: {str(e)}")
            self.logger.error(f"保存模板失败: {str(e)}")
            
    def _on_delete_template(self):
        """删除模板按钮点击事件"""
        # 获取当前选择的模板
        current_template = self.template_combo.currentText()
        if not current_template:
            self.status_label.setText("状态: 错误 - 未选择模板")
            return
            
        # 删除模板目录
        template_path = self.template_dir / current_template
        try:
            if template_path.exists():
                import shutil
                shutil.rmtree(template_path)
                
            # 更新模板下拉列表
            self._update_template_combo()
            
            # 清空界面
            self._on_new_template()
            
            self.status_label.setText(f"状态: 已删除模板 {current_template}")
        except Exception as e:
            self.status_label.setText(f"状态: 错误 - 删除模板失败: {str(e)}")
            self.logger.error(f"删除模板失败: {str(e)}")
            
    def _update_template_combo(self):
        """更新模板下拉列表"""
        self.template_combo.clear()
        
        # 遍历模板目录
        for frame_dir in self.template_dir.iterdir():
            if frame_dir.is_dir():
                for template_dir in frame_dir.iterdir():
                    if template_dir.is_dir():
                        config_file = template_dir / "config.yml"
                        if config_file.exists():
                            with open(config_file, 'r', encoding='utf-8') as f:
                                config = yaml.safe_load(f)
                                display_name = f"{config['frame_id']}/{config['template_id']} - {config['template_name']}"
                                self.template_combo.addItem(display_name, str(template_dir))
                                
    def _load_image(self, image_path):
        """加载图片到界面"""
        try:
            # 使用OpenCV读取图片
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                raise ValueError("无法加载图片")
                
            # 转换为QPixmap并显示
            height, width, channel = self.current_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(self.current_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
        except Exception as e:
            self.status_label.setText(f"状态: 错误 - 加载图片失败: {str(e)}")
            self.logger.error(f"加载图片失败: {str(e)}")
            
    def _update_button_states(self, enabled):
        """更新按钮状态"""
        self.open_button.setEnabled(enabled)
        self.save_button.setEnabled(enabled)
        self.delete_button.setEnabled(enabled)
        self.crop_button.setEnabled(enabled)
        
    def _apply_hsv_crop(self):
        """应用HSV分割到裁剪区域"""
        if not hasattr(self, 'current_image') or self.current_image is None:
            return
            
        try:
            # 保存原始图像
            self.original_image = self.current_image.copy()
            
            h_upper = int(self.h_upper_edit.text().strip())
            s_upper = int(self.s_upper_edit.text().strip())
            v_upper = int(self.v_upper_edit.text().strip())
            h_lower = int(self.h_lower_edit.text().strip())
            s_lower = int(self.s_lower_edit.text().strip())
            v_lower = int(self.v_lower_edit.text().strip())
            
            lower = np.array([h_lower, s_lower, v_lower])
            upper = np.array([h_upper, s_upper, v_upper])
            
            # 如果有裁剪区域，先裁剪图像
            if self.crop_text.text().strip():
                x1, y1, x2, y2 = map(int, self.crop_text.text().strip().split(','))
                cropped_img = self.current_image[y1:y2, x1:x2]
            else:
                cropped_img = self.current_image
                
            hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(cropped_img, cropped_img, mask=mask)
            
            self._display_crop_preview(result)
            self.apply_crop_button.setText('恢复裁剪区域')
            self.apply_crop_button.clicked.disconnect()
            self.apply_crop_button.clicked.connect(lambda: self._restore_original(crop=True))
            
        except Exception as e:
            self.status_label.setText(f"状态: 错误 - {str(e)}")
            self.logger.error(f"HSV分割失败: {str(e)}")
            
    def _apply_hsv_original(self):
        """应用HSV分割到整个图像"""
        if not hasattr(self, 'current_image') or self.current_image is None:
            return
            
        try:
            # 保存原始图像
            self.original_image = self.current_image.copy()
            
            h_upper = int(self.h_upper_edit.text().strip())
            s_upper = int(self.s_upper_edit.text().strip())
            v_upper = int(self.v_upper_edit.text().strip())
            h_lower = int(self.h_lower_edit.text().strip())
            s_lower = int(self.s_lower_edit.text().strip())
            v_lower = int(self.v_lower_edit.text().strip())
            
            lower = np.array([h_lower, s_lower, v_lower])
            upper = np.array([h_upper, s_upper, v_upper])
            
            hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(self.current_image, self.current_image, mask=mask)
            
            self._display_image(result)
            self.apply_original_button.setText('恢复完整图像')
            self.apply_original_button.clicked.disconnect()
            self.apply_original_button.clicked.connect(lambda: self._restore_original(crop=False))
            
        except Exception as e:
            self.status_label.setText(f"状态: 错误 - {str(e)}")
            self.logger.error(f"HSV分割失败: {str(e)}")
            
    def _display_image(self, image):
        """显示处理后的图像"""
        height, width = image.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))
        
    def _display_crop_preview(self, image):
        """显示裁剪预览图像"""
        height, width = image.shape[:2]
        bytes_per_line = 3 * width
        # 将图像转换为RGB格式
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.crop_preview_label.setPixmap(pixmap.scaled(
            self.crop_preview_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))
        
    def _restore_original(self, crop=False):
        """恢复原始图像
        Args:
            crop: 是否恢复裁剪模式
        """
        if hasattr(self, 'original_image'):
            if crop and hasattr(self, 'current_image') and self.crop_text.text().strip():
                x1, y1, x2, y2 = map(int, self.crop_text.text().strip().split(','))
                cropped_img = self.original_image[y1:y2, x1:x2]
                self._display_crop_preview(cropped_img)
            else:
                self._display_image(self.original_image)
            
        if crop:
            self.apply_crop_button.setText('分割裁剪区域')
            self.apply_crop_button.clicked.disconnect()
            self.apply_crop_button.clicked.connect(self._apply_hsv_crop)
        else:
            self.apply_original_button.setText('分割完整图像')
            self.apply_original_button.clicked.disconnect()
            self.apply_original_button.clicked.connect(self._apply_hsv_original)
    
    def _on_image_clicked(self, event):
        """图片点击事件处理"""
        if not hasattr(self, 'current_image') or self.current_image is None:
            return
            
        # 获取点击位置
        x = event.pos().x()
        y = event.pos().y()
        
        # 计算实际图像坐标（考虑缩放和居中）
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return
            
        # 计算实际图像坐标
        img_width = self.current_image.shape[1]
        img_height = self.current_image.shape[0]
        
        # 获取QLabel尺寸
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        
        # 计算缩放比例和偏移
        scale = min(label_width / img_width, label_height / img_height)
        offset_x = (label_width - img_width * scale) / 2
        offset_y = (label_height - img_height * scale) / 2
        
        # 计算实际图像坐标
        img_x = int((x - offset_x) / scale)
        img_y = int((y - offset_y) / scale)
        
        # 确保坐标在图像范围内
        if 0 <= img_x < img_width and 0 <= img_y < img_height:
            # 获取BGR颜色
            bgr = self.current_image[img_y, img_x]
            
            # 转换为RGB
            rgb = (bgr[2], bgr[1], bgr[0])
            
            # 转换为HSV
            hsv = cv2.cvtColor(np.array([[bgr]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
            
            # 更新状态栏
            self.status_label.setText(
                f"状态: 点击位置 ({img_x}, {img_y}) | "
                f"RGB: ({int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])}) | "
                f"HSV: ({int(hsv[0])}, {int(hsv[1])}, {int(hsv[2])})"
            )
            
            # 处理裁剪模式下的点击
            if self.is_crop_mode:
                if not hasattr(self, 'start_pos') or self.start_pos is None:
                    self.start_pos = (img_x, img_y)
                    self.status_label.setText("状态: 裁剪模式 - 请点击图像选择终点")
                elif not hasattr(self, 'end_pos') or self.end_pos is None:
                    self.end_pos = (img_x, img_y)
                    # 更新预览框
                    if hasattr(self, 'current_image') and self.current_image is not None:
                        x1, y1 = self.start_pos
                        x2, y2 = self.end_pos
                        cropped_img = self.current_image[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
                        height, width, _ = cropped_img.shape
                        bytes_per_line = 3 * width
                        rgb_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                        q_img = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(q_img)
                        self.crop_preview_label.setPixmap(pixmap.scaled(
                            self.crop_preview_label.size(), 
                            Qt.KeepAspectRatio, 
                            Qt.SmoothTransformation
                        ))
                    self._toggle_crop_mode()
        
    def resizeEvent(self, event):
        """窗口大小改变时重新缩放图片"""
        super().resizeEvent(event)
        if hasattr(self, 'current_image') and self.current_image is not None:
            height, width, _ = self.current_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(self.current_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
        
    def _on_new_template(self):
        """新建模板按钮点击事件"""
        # 设置默认内容
        self.frame_id_edit.setText("default_frame")
        self.template_id_edit.setText("default_template")
        self.template_name_edit.setText("新模板")
        self.h_upper_edit.setText("179")
        self.s_upper_edit.setText("255")
        self.v_upper_edit.setText("255")
        self.h_lower_edit.setText("0")
        self.s_lower_edit.setText("0")
        self.v_lower_edit.setText("0")
        self.crop_text.clear()
        
        # 更新状态和按钮
        self.status_label.setText("状态: 新建模板准备就绪")
        self._update_button_states(True)
        
        # 加载默认图片
        default_img = np.zeros((300, 300, 3), dtype=np.uint8)
        self.current_image = default_img
        height, width, channel = default_img.shape
        bytes_per_line = 3 * width
        q_img = QImage(default_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))