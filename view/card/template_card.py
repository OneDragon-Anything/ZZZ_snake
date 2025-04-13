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
from model.template.template import Template
from model.template.template_manage import TemplateManager

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
        
        # 新增形状和差异设置
        shape_layout = QHBoxLayout()
        shape_layout.addWidget(BodyLabel('形状类型:'))
        self.shape_combo = ComboBox()
        self.shape_combo.addItems(['circle', 'rectangle', 'any'])
        self.shape_combo.setCurrentText('circle')
        shape_layout.addWidget(self.shape_combo)
        left_layout.addLayout(shape_layout)
        
        self.area_diff_edit = self._create_input_row('面积差异:', left_layout)
        self.area_diff_edit.setText('0.3')
        self.shape_diff_edit = self._create_input_row('形状差异:', left_layout)
        self.shape_diff_edit.setText('0.3')
        
        left_layout.addSpacing(15)
        # HSV颜色设置
        hsv_color_layout = QHBoxLayout()
        hsv_color_layout.addWidget(BodyLabel('HSV颜色:'))
        self.h_hsv_color_edit = LineEdit()
        self.h_hsv_color_edit.setPlaceholderText('H(0-179)')
        self.s_hsv_color_edit = LineEdit()
        self.s_hsv_color_edit.setPlaceholderText('S(0-255)')
        self.v_hsv_color_edit = LineEdit()
        self.v_hsv_color_edit.setPlaceholderText('V(0-255)')
        hsv_color_layout.addWidget(self.h_hsv_color_edit)
        hsv_color_layout.addWidget(self.s_hsv_color_edit)
        hsv_color_layout.addWidget(self.v_hsv_color_edit)
        left_layout.addLayout(hsv_color_layout)
        
        # HSV色差设置
        hsv_diff_layout = QHBoxLayout()
        hsv_diff_layout.addWidget(BodyLabel('HSV色差:'))
        self.h_hsv_diff_edit = LineEdit()
        self.h_hsv_diff_edit.setPlaceholderText('H(0-179)')
        self.s_hsv_diff_edit = LineEdit()
        self.s_hsv_diff_edit.setPlaceholderText('S(0-255)')
        self.v_hsv_diff_edit = LineEdit()
        self.v_hsv_diff_edit.setPlaceholderText('V(0-255)')
        hsv_diff_layout.addWidget(self.h_hsv_diff_edit)
        hsv_diff_layout.addWidget(self.s_hsv_diff_edit)
        hsv_diff_layout.addWidget(self.v_hsv_diff_edit)
        left_layout.addLayout(hsv_diff_layout)
        
        # 8. HSV应用按钮
        hsv_apply_layout = QHBoxLayout()
        self.apply_template_button = PrimaryPushButton('过滤模板图像')
        self.apply_original_button = PrimaryPushButton('过滤完整图像')
        self.apply_template_button.clicked.connect(self._apply_hsv_crop)
        self.apply_original_button.clicked.connect(self._apply_hsv_original)
        hsv_apply_layout.addWidget(self.apply_template_button)
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
        self.template_preview_label = QLabel()
        self.template_preview_label.setFixedSize(200, 200)
        self.template_preview_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.template_preview_label)
        
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
        
        # 初始化模板管理器
        template_dir = Path(__file__).parent.parent.parent / "templates"
        self.template_manager = TemplateManager(template_dir)
        
        # 连接按钮信号
        self.new_button.clicked.connect(self._on_new_template)
        self.open_button.clicked.connect(self._on_read_frame)
        self.save_button.clicked.connect(self._on_save_template)
        self.delete_button.clicked.connect(self._on_delete_template)
        self.template_combo.currentIndexChanged.connect(self._on_template_selected)
        self.position_button.clicked.connect(self._on_position)
        
        # 初始禁用按钮
        self._update_button_states(False)
        
        # 更新模板列表
        self._update_template_combo()
    
    def _create_input_row(self, label_text, parent_layout):
        """创建带标签的输入行"""
        row_layout = QHBoxLayout()
        label = BodyLabel(label_text)
        edit = LineEdit()
        
        # 如果是HSV色调输入框，添加验证器
        if label_text == 'H(0-179):':
            from PyQt5.QtGui import QIntValidator
            validator = QIntValidator(0, 179)
            edit.setValidator(validator)
            
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
        
        # 获取新增的形状和差异设置
        shape_type = self.shape_combo.currentText()
        area_diff = float(self.area_diff_edit.text().strip())
        shape_diff = float(self.shape_diff_edit.text().strip())
        
        # 验证HSV输入
        try:
            h_hsv_color = int(self.h_hsv_color_edit.text().strip())
            s_hsv_color = int(self.s_hsv_color_edit.text().strip())
            v_hsv_color = int(self.v_hsv_color_edit.text().strip())
            h_hsv_diff = int(self.h_hsv_diff_edit.text().strip())
            s_hsv_diff = int(self.s_hsv_diff_edit.text().strip())
            v_hsv_diff = int(self.v_hsv_diff_edit.text().strip())
            
            if not (0 <= h_hsv_color <= 179 and 0 <= h_hsv_diff <= 179):
                raise ValueError("H值必须在0-179之间")
            if not (0 <= s_hsv_color <= 255 and 0 <= s_hsv_diff <= 255 and 
                    0 <= v_hsv_color <= 255 and 0 <= v_hsv_diff <= 255):
                raise ValueError("S和V值必须在0-255之间")
                
            hsv_color = f"{h_hsv_color},{s_hsv_color},{v_hsv_color}"
            hsv_diff = f"{h_hsv_diff},{s_hsv_diff},{v_hsv_diff}"
        except ValueError as e:
            self.status_label.setText(f"状态: 错误 - {str(e)}")
            return
            
        crop_area = self.crop_text.text().strip()
        
        # 验证必填字段
        if not all([frame_id, template_id, template_name]):
            self.status_label.setText("状态: 错误 - 画面ID、模板ID和名称不能为空")
            return
            
        try:
            # 创建模板对象
            template = Template(
                frame_id=frame_id,
                template_id=template_id,
                template_name=template_name,
                hsv_color=hsv_color,
                hsv_diff=hsv_diff,
                crop_area=crop_area,
                area_diff=area_diff,  # 新增
                shape_type=shape_type,  # 新增
                shape_diff=shape_diff  # 新增
            )
            
            # 获取要保存的图片
            if hasattr(self, 'template_image') and self.template_image is not None:
                save_img = self.template_image.copy()  # 直接使用已保存的模板图片
            elif crop_area:
                x1, y1, x2, y2 = map(int, crop_area.split(','))
                save_img = self.current_image[y1:y2, x1:x2].copy()  # 如果没有模板图片则使用裁剪图像
            else:
                save_img = self.current_image.copy()  # 使用完整图片
            
            # 使用Template类的save方法保存模板
            template.save(self.template_manager.template_dir, save_img)
                
            # 临时断开信号连接
            self.template_combo.currentIndexChanged.disconnect()
            
            # 更新模板下拉列表
            self._update_template_combo()
            
            # 重新连接信号
            self.template_combo.currentIndexChanged.connect(self._on_template_selected)
            
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
            
        try:
            # 从显示文本中提取frame_id和template_id
            frame_id, rest = current_template.split('/', 1)
            template_id = rest.split(' - ')[0]
            
            # 删除模板
            if self.template_manager.delete_template(frame_id, template_id):
                # 更新模板下拉列表
                self._update_template_combo()
                
                # 清空界面
                self._on_new_template()
                
                self.status_label.setText(f"状态: 已删除模板 {current_template}")
            else:
                self.status_label.setText("状态: 错误 - 模板不存在")
        except Exception as e:
            self.status_label.setText(f"状态: 错误 - 删除模板失败: {str(e)}")
            self.logger.error(f"删除模板失败: {str(e)}")
            
    def _update_template_combo(self):
        """更新模板下拉列表"""
        # 保存当前选中的模板ID
        current_template_id = None
        if self.template_combo.currentIndex() >= 0 and hasattr(self, 'all_templates'):
            current_template = self.all_templates[self.template_combo.currentIndex()]
            current_template_id = f"{current_template.frame_id}/{current_template.template_id}"

        self.template_combo.clear()
        self.all_templates = self.template_manager.get_all_templates()
        
        for template in self.all_templates:
            display_name = f"{template.frame_id}/{template.template_id} - {template.template_name}"
            self.template_combo.addItem(display_name)
            
        # 恢复之前选中的模板
        if current_template_id:
            for i, template in enumerate(self.all_templates):
                if f"{template.frame_id}/{template.template_id}" == current_template_id:
                    self.template_combo.setCurrentIndex(i)
                    break

    def _on_template_selected(self, index):
        """模板选择事件处理"""
        if index < 0:
            return
            
        try:
            # 直接从all_templates获取模板对象
            if index >= len(self.all_templates):
                self.status_label.setText("状态: 错误 - 无效的模板索引")
                return
                
            template = self.all_templates[index]
            if not template:
                self.status_label.setText("状态: 错误 - 无效的模板数据")
                return
                
            # 更新界面显示
            self.frame_id_edit.setText(template.frame_id)
            self.template_id_edit.setText(template.template_id)
            self.template_name_edit.setText(template.template_name)
            
            # 更新新增的形状和差异设置
            self.shape_combo.setCurrentText(template.shape_type)
            self.area_diff_edit.setText(str(template.area_diff))
            self.shape_diff_edit.setText(str(template.shape_diff))
            
            # 设置HSV值
            h_hsv_color, s_hsv_color, v_hsv_color = template.hsv_color.split(',')
            h_hsv_diff, s_hsv_diff, v_hsv_diff = template.hsv_diff.split(',')
            self.h_hsv_color_edit.setText(h_hsv_color)
            self.s_hsv_color_edit.setText(s_hsv_color)
            self.v_hsv_color_edit.setText(v_hsv_color)
            self.h_hsv_diff_edit.setText(h_hsv_diff)
            self.s_hsv_diff_edit.setText(s_hsv_diff)
            self.v_hsv_diff_edit.setText(v_hsv_diff)
            
            # 设置模板图像
            if template.crop_area:
                self.crop_text.setText(template.crop_area)
            else:
                self.crop_text.clear()
            
            # 加载模板图片
            template_path = self.template_manager.template_dir / template.frame_id / template.template_id
            image_path = template_path / "template.png"
            if image_path.exists():
                # 使用临时变量存储模板图片，不影响当前主图像
                self.template_image = cv2.imread(str(image_path))
                if self.template_image is not None:
                    # 直接显示到裁剪预览框
                    self._display_template_preview(self.template_image)
                    self.status_label.setText(f"状态: 已加载模板 {template.template_name}")
                    # 启用按钮
                    self._update_button_states(True)
                else:
                    raise ValueError("无法加载图片")
            else:
                self.status_label.setText(f"状态: 警告 - 模板图片不存在")
                self.logger.warning(f"模板图片不存在: {image_path}")
                
        except Exception as e:
            self.status_label.setText(f"状态: 错误 - 加载模板失败: {str(e)}")
            self.logger.error(f"加载模板失败: {str(e)}")
    
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
        """应用HSV过滤到模板图像"""
        if not hasattr(self, 'template_image') or self.template_image is None:
            return
            
        try:
            h_hsv_color = int(self.h_hsv_color_edit.text().strip())
            s_hsv_color = int(self.s_hsv_color_edit.text().strip())
            v_hsv_color = int(self.v_hsv_color_edit.text().strip())
            h_hsv_diff = int(self.h_hsv_diff_edit.text().strip())
            s_hsv_diff = int(self.s_hsv_diff_edit.text().strip())
            v_hsv_diff = int(self.v_hsv_diff_edit.text().strip())
            
            # 使用TemplateManager的filter_image_by_hsv方法
            result = self.template_manager.filter_image_by_hsv(
                self.template_image,
                (h_hsv_color, s_hsv_color, v_hsv_color),
                (h_hsv_diff, s_hsv_diff, v_hsv_diff)
            )
            
            self._display_template_preview(result)
            self.apply_template_button.setText('恢复模板图像')
            self.apply_template_button.clicked.disconnect()
            self.apply_template_button.clicked.connect(lambda: self._restore_original(crop=True))
            
        except Exception as e:
            self.status_label.setText(f"状态: 错误 - {str(e)}")
            self.logger.error(f"HSV过滤失败: {str(e)}")
            
    def _apply_hsv_original(self):
        """应用HSV过滤到整个图像"""
        if not hasattr(self, 'current_image') or self.current_image is None:
            return
            
        try:
            # 保存原始图像
            self.original_image = self.current_image.copy()
            
            h_hsv_color = int(self.h_hsv_color_edit.text().strip())
            s_hsv_color = int(self.s_hsv_color_edit.text().strip())
            v_hsv_color = int(self.v_hsv_color_edit.text().strip())
            h_hsv_diff = int(self.h_hsv_diff_edit.text().strip())
            s_hsv_diff = int(self.s_hsv_diff_edit.text().strip())
            v_hsv_diff = int(self.v_hsv_diff_edit.text().strip())
            
            # 使用TemplateManager的filter_image_by_hsv方法
            result = self.template_manager.filter_image_by_hsv(
                self.current_image,
                (h_hsv_color, s_hsv_color, v_hsv_color),
                (h_hsv_diff, s_hsv_diff, v_hsv_diff)
            )
            
            self._display_image(result)
            self.apply_original_button.setText('恢复完整图像')
            self.apply_original_button.clicked.disconnect()
            self.apply_original_button.clicked.connect(lambda: self._restore_original(crop=False))
            
        except Exception as e:
            self.status_label.setText(f"状态: 错误 - {str(e)}")
            self.logger.error(f"HSV过滤失败: {str(e)}")
            
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
        
    def _display_template_preview(self, image):
        """显示模板预览图像"""
        height, width = image.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        self.template_preview_label.setPixmap(pixmap.scaled(
            self.template_preview_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))
        
    def _restore_original(self, crop=False):
        """恢复原始图像
        Args:
            crop: 是否恢复裁剪模式
        """
        if crop:
            if hasattr(self, 'template_image') and self.template_image is not None:
                # 直接使用已保存的模板图片
                rgb_img = self.template_image
                self._display_template_preview(rgb_img)
            elif hasattr(self, 'current_image') and self.crop_text.text().strip():
                # 如果没有模板图片但有裁剪区域
                x1, y1, x2, y2 = map(int, self.crop_text.text().strip().split(','))
                cropped_img = self.current_image[y1:y2, x1:x2]
                self._display_template_preview(cropped_img)
        elif hasattr(self, 'original_image'):
            # 恢复完整图像时才需要original_image
            self._display_image(self.original_image)
            
        if crop:
            self.apply_template_button.setText('过滤模板图像')
            self.apply_template_button.clicked.disconnect()
            self.apply_template_button.clicked.connect(self._apply_hsv_crop)
        else:
            self.apply_original_button.setText('过滤完整图像')
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
                        # 裁剪并保存到template_image
                        self.template_image = self.current_image[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)].copy()
                        # 显示预览
                        self._display_template_preview(self.template_image)
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
        # 设置新增控件的默认值
        self.shape_combo.setCurrentText('circle')
        self.area_diff_edit.setText('0.3')
        self.shape_diff_edit.setText('0.3')

        self.h_hsv_color_edit.setText("179")
        self.s_hsv_color_edit.setText("255")
        self.v_hsv_color_edit.setText("255")
        self.h_hsv_diff_edit.setText("0")
        self.s_hsv_diff_edit.setText("0")
        self.v_hsv_diff_edit.setText("0")
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

    def _on_position(self):
        """定位按钮点击事件"""
        if not hasattr(self, 'current_image') or self.current_image is None:
            self.status_label.setText("状态: 错误 - 请先加载图片")
            return
            
        # 获取当前选中的模板
        index = self.template_combo.currentIndex()
        if index < 0 or index >= len(self.all_templates):
            self.status_label.setText("状态: 错误 - 请先选择模板")
            return
            
        template = self.all_templates[index]
        
        try:
            # 在图片中查找模板
            matches = self.template_manager.find_objects_by_features(template, self.current_image)
            
            if not matches:
                self.status_label.setText("状态: 未找到匹配结果")
                return
                
            # 在原图上绘制匹配结果
            result_image = self.current_image.copy()
            for x, y, angle, score in matches:
                # 绘制十字标记
                size = 20
                color = (0, 255, 0)  # 绿色
                thickness = 2
                cv2.line(result_image, (x-size, y), (x+size, y), color, thickness)
                cv2.line(result_image, (x, y-size), (x, y+size), color, thickness)
                
                # 显示匹配分数
                text = f"{score:.2f}"
                cv2.putText(result_image, text, (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
                
                # 如果有旋转角度，显示角度
                if angle != 0:
                    angle_text = f"{angle:.1f}°"
                    cv2.putText(result_image, angle_text, (x+10, y+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
            
            # 显示结果图片
            self._display_image(result_image)
            self.status_label.setText(f"状态: 找到 {len(matches)} 个匹配结果")
            
        except Exception as e:
            self.status_label.setText(f"状态: 错误 - 模板匹配失败: {str(e)}")
            self.logger.error(f"模板匹配失败: {str(e)}")