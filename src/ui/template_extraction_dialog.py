import cv2
import numpy as np
import os
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
                           QGroupBox, QFormLayout, QComboBox, QSlider, QCheckBox,
                           QScrollArea, QLineEdit, QRadioButton, QButtonGroup,
                           QMessageBox, QSplitter, QFrame)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QRect

class RectangleSelector:
    """Helper class to select a rectangle region in an image"""
    
    def __init__(self, image):
        self.image = image.copy()
        self.original = image.copy()
        self.start_point = None
        self.end_point = None
        self.selecting = False
        self.selected_rect = None
    
    def on_mouse_event(self, event, x, y, flags, param):
        """Handle mouse events for rectangle selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selecting:
                # Create a copy of the original image
                self.image = self.original.copy()
                # Draw a rectangle
                cv2.rectangle(self.image, self.start_point, (x, y), (0, 255, 0), 2)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.end_point = (x, y)
            self.selecting = False
            
            # Draw the final rectangle
            cv2.rectangle(self.image, self.start_point, self.end_point, (0, 255, 0), 2)
            
            # Normalize the rectangle coordinates
            x1, y1 = min(self.start_point[0], self.end_point[0]), min(self.start_point[1], self.end_point[1])
            x2, y2 = max(self.start_point[0], self.end_point[0]), max(self.start_point[1], self.end_point[1])
            
            self.selected_rect = (x1, y1, x2 - x1, y2 - y1)  # (x, y, w, h)
    
    def get_selected_region(self):
        """Return the selected rectangle"""
        return self.selected_rect
    
    def get_drawn_image(self):
        """Return the image with selection rectangle"""
        return self.image


class TemplateExtractionDialog(QDialog):
    """Dialog for extracting sign templates from an image"""
    
    extraction_complete = pyqtSignal(bool)
    
    def __init__(self, image_path, template_manager, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Template Extraction")
        self.resize(1000, 700)
        
        self.image_path = image_path
        self.template_manager = template_manager
        self.image = cv2.imread(image_path)
        
        if self.image is None:
            QMessageBox.critical(self, "Error", "Failed to load the image.")
            self.reject()
            return
            
        # Create layout
        layout = QVBoxLayout(self)
        
        # Add title
        title_label = QLabel(f"Extract Templates from: {os.path.basename(image_path)}")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)
        
        # Create splitter for image and settings
        splitter = QSplitter(Qt.Horizontal)
        
        # Image view (left side)
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        
        # Show image
        self.image_label = QLabel()
        self.update_image_display(self.image)
        
        # Create a scroll area for the image
        scroll = QScrollArea()
        scroll.setWidget(self.image_label)
        scroll.setWidgetResizable(True)
        image_layout.addWidget(scroll)
        
        # Add image controls
        controls_layout = QHBoxLayout()
        
        self.auto_detect_btn = QPushButton("Auto-Detect Signs")
        self.manual_select_btn = QPushButton("Select Region")
        self.reset_btn = QPushButton("Reset")
        
        controls_layout.addWidget(self.auto_detect_btn)
        controls_layout.addWidget(self.manual_select_btn)
        controls_layout.addWidget(self.reset_btn)
        
        image_layout.addLayout(controls_layout)
        
        # Settings panel (right side)
        settings_panel = QWidget()
        settings_layout = QVBoxLayout(settings_panel)
        
        # Template info group box
        template_group = QGroupBox("Template Properties")
        template_form = QFormLayout(template_group)
        
        # Sign category selection
        self.category_combo = QComboBox()
        self.category_combo.addItems(["Regulatory", "Warning", "Information"])
        template_form.addRow("Sign Category:", self.category_combo)
        
        # Sign type
        self.sign_type_combo = QComboBox()
        self.sign_type_combo.setEditable(True)
        self.populate_sign_types()
        template_form.addRow("Sign Type:", self.sign_type_combo)
        
        # Template name
        self.template_name = QLineEdit()
        self.template_name.setText(f"template_{os.path.splitext(os.path.basename(image_path))[0]}")
        template_form.addRow("Template Name:", self.template_name)
        
        settings_layout.addWidget(template_group)
        
        # Processing options group box
        processing_group = QGroupBox("Processing Options")
        processing_form = QVBoxLayout(processing_group)
        
        # Preprocessing options
        self.preprocess_check = QCheckBox("Apply preprocessing")
        self.preprocess_check.setChecked(True)
        processing_form.addWidget(self.preprocess_check)
        
        # Size options
        size_layout = QHBoxLayout()
        self.resize_64_radio = QRadioButton("64x64")
        self.resize_96_radio = QRadioButton("96x96")
        self.resize_128_radio = QRadioButton("128x128")
        self.resize_original_radio = QRadioButton("Original")
        
        # Group radio buttons
        size_group = QButtonGroup(self)
        size_group.addButton(self.resize_64_radio)
        size_group.addButton(self.resize_96_radio)
        size_group.addButton(self.resize_128_radio)
        size_group.addButton(self.resize_original_radio)
        
        # Default to 64x64
        self.resize_64_radio.setChecked(True)
        
        size_layout.addWidget(QLabel("Size:"))
        size_layout.addWidget(self.resize_64_radio)
        size_layout.addWidget(self.resize_96_radio)
        size_layout.addWidget(self.resize_128_radio)
        size_layout.addWidget(self.resize_original_radio)
        
        processing_form.addLayout(size_layout)
        
        # Add augmentation options
        self.augment_check = QCheckBox("Generate augmented variants")
        processing_form.addWidget(self.augment_check)
        
        settings_layout.addWidget(processing_group)
        
        # Preview section
        preview_group = QGroupBox("Template Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel("No template selected")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(150, 150)
        self.preview_label.setFrameShape(QFrame.StyledPanel)
        preview_layout.addWidget(self.preview_label)
        
        settings_layout.addWidget(preview_group)
        
        # Add extraction status
        self.status_label = QLabel("Ready to extract template")
        settings_layout.addWidget(self.status_label)
        
        # Add buttons
        buttons_layout = QHBoxLayout()
        
        self.extract_btn = QPushButton("Extract Template")
        self.extract_btn.setEnabled(False)
        self.extract_multiple_btn = QPushButton("Extract Multiple...")
        self.cancel_btn = QPushButton("Cancel")
        
        buttons_layout.addWidget(self.extract_btn)
        buttons_layout.addWidget(self.extract_multiple_btn)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.cancel_btn)
        
        settings_layout.addLayout(buttons_layout)
        settings_layout.addStretch()
        
        # Add panels to splitter
        splitter.addWidget(image_panel)
        splitter.addWidget(settings_panel)
        
        # Set initial splitter sizes
        splitter.setSizes([700, 300])
        
        layout.addWidget(splitter)
        
        # Connect signals
        self.connect_signals()
        
        # Initialize rectangle selector
        self.rect_selector = RectangleSelector(self.image)
        self.selected_rect = None
    
    def connect_signals(self):
        """Connect UI signals to slots"""
        self.auto_detect_btn.clicked.connect(self.auto_detect_signs)
        self.manual_select_btn.clicked.connect(self.start_manual_selection)
        self.reset_btn.clicked.connect(self.reset_selection)
        
        self.extract_btn.clicked.connect(self.extract_template)
        self.extract_multiple_btn.clicked.connect(self.extract_multiple)
        self.cancel_btn.clicked.connect(self.reject)
        
        self.category_combo.currentIndexChanged.connect(self.populate_sign_types)
    
    def update_image_display(self, image):
        """Update the image display with the given image"""
        if image is not None:
            # Convert OpenCV image to QPixmap
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            
            self.image_label.setPixmap(pixmap)
            self.image_label.setFixedSize(pixmap.width(), pixmap.height())
    
    def populate_sign_types(self):
        """Populate sign types based on selected category"""
        self.sign_type_combo.clear()
        
        category = self.category_combo.currentText().lower()
        
        # Get subdirectories for the selected category
        if category == "regulatory":
            dir_path = self.template_manager.regulatory_dir
        elif category == "warning":
            dir_path = self.template_manager.warning_dir
        else:
            dir_path = self.template_manager.information_dir
            
        if os.path.exists(dir_path):
            sign_types = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
            for sign_type in sorted(sign_types):
                self.sign_type_combo.addItem(sign_type)
    
    def auto_detect_signs(self):
        """Auto-detect signs in the image"""
        # This would use your sign detector to find signs automatically
        # For now, display a message that it's not implemented
        QMessageBox.information(
            self, "Auto-detection",
            "Automatic sign detection will be implemented.\n"
            "Currently, please use manual selection."
        )
    
    def start_manual_selection(self):
        """Start manual region selection mode"""
        # Create a copy of the image for selection
        selection_img = self.image.copy()
        
        # Set up window for rectangle selection
        cv2.namedWindow("Select Sign Region")
        cv2.setMouseCallback("Select Sign Region", self.rect_selector.on_mouse_event)
        
        # Instructions
        cv2.putText(selection_img, "Click and drag to select sign region. Press 'c' to cancel, 's' to save selection.",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show the image and wait for selection
        while True:
            cv2.imshow("Select Sign Region", self.rect_selector.get_drawn_image())
            key = cv2.waitKey(1) & 0xFF
            
            # Press 's' to save selection or 'c' to cancel
            if key == ord('s'):
                self.selected_rect = self.rect_selector.get_selected_region()
                cv2.destroyAllWindows()
                break
            elif key == ord('c') or key == 27:  # ESC key
                cv2.destroyAllWindows()
                return
        
        # Update UI with the selection
        if self.selected_rect:
            # Draw rectangle on the image
            x, y, w, h = self.selected_rect
            display_img = self.image.copy()
            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            self.update_image_display(display_img)
            
            # Extract and show preview
            self.update_preview()
            
            # Enable extract button
            self.extract_btn.setEnabled(True)
            
            # Update status
            self.status_label.setText("Region selected. Ready to extract.")
    
    def update_preview(self):
        """Update the template preview"""
        if self.selected_rect:
            x, y, w, h = self.selected_rect
            
            # Extract the region
            if (x >= 0 and y >= 0 and 
                x + w <= self.image.shape[1] and
                y + h <= self.image.shape[0]):
                
                roi = self.image[y:y+h, x:x+w]
                
                # Determine preview size
                preview_size = (64, 64)
                if self.resize_96_radio.isChecked():
                    preview_size = (96, 96)
                elif self.resize_128_radio.isChecked():
                    preview_size = (128, 128)
                elif self.resize_original_radio.isChecked():
                    preview_size = (roi.shape[1], roi.shape[0])
                
                # Apply preprocessing if checked
                if self.preprocess_check.isChecked():
                    # Simple preprocessing (can be expanded)
                    roi = cv2.GaussianBlur(roi, (3, 3), 0)
                    roi = cv2.resize(roi, preview_size)
                else:
                    roi = cv2.resize(roi, preview_size)
                
                # Update preview
                height, width, channel = roi.shape
                bytes_per_line = 3 * width
                q_img = QImage(roi.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(q_img)
                
                self.preview_label.setPixmap(pixmap)
                self.preview_label.setFixedSize(pixmap.width(), pixmap.height())
    
    def reset_selection(self):
        """Reset the selection"""
        self.selected_rect = None
        self.update_image_display(self.image)
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText("No template selected")
        self.extract_btn.setEnabled(False)
        self.status_label.setText("Ready to extract template")
    
    def extract_template(self):
        """Extract the selected region as a template"""
        if not self.selected_rect:
            QMessageBox.warning(self, "No Selection", "Please select a region first.")
            return
            
        # Get template properties
        category = self.category_combo.currentText().lower()
        sign_type = self.sign_type_combo.currentText()
        
        if not sign_type:
            QMessageBox.warning(self, "Missing Information", "Please enter a sign type.")
            return
            
        # Determine save directory
        if category == "regulatory":
            save_dir = os.path.join(self.template_manager.regulatory_dir, sign_type)
        elif category == "warning":
            save_dir = os.path.join(self.template_manager.warning_dir, sign_type)
        else:
            save_dir = os.path.join(self.template_manager.information_dir, sign_type)
            
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract the region
        x, y, w, h = self.selected_rect
        roi = self.image[y:y+h, x:x+w]
        
        # Determine template size
        template_size = (64, 64)
        if self.resize_96_radio.isChecked():
            template_size = (96, 96)
        elif self.resize_128_radio.isChecked():
            template_size = (128, 128)
        elif self.resize_original_radio.isChecked():
            template_size = (roi.shape[1], roi.shape[0])
        
        # Apply preprocessing if checked
        if self.preprocess_check.isChecked():
            # Simple preprocessing (can be expanded)
            roi = cv2.GaussianBlur(roi, (3, 3), 0)
            
        # Resize to standard template size
        if not self.resize_original_radio.isChecked():
            roi = cv2.resize(roi, template_size)
        
        # Generate filename
        template_name = self.template_name.text()
        if not template_name:
            template_name = f"template_{os.path.splitext(os.path.basename(self.image_path))[0]}"
            
        filename = f"{sign_type}_{template_name}.jpg"
        save_path = os.path.join(save_dir, filename)
        
        # Check if file already exists
        if os.path.exists(save_path):
            confirm = QMessageBox.question(
                self, "File Exists", 
                f"The file {filename} already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if confirm != QMessageBox.Yes:
                return
                
        # Save the template
        cv2.imwrite(save_path, roi)
        
        # Generate augmented variants if checked
        if self.augment_check.isChecked():
            self.generate_augmented_templates(roi, save_dir, sign_type, template_name)
        
        QMessageBox.information(
            self, "Template Extracted",
            f"Template saved to {save_path}"
        )
        
        # Update status
        self.status_label.setText(f"Template extracted to {save_path}")
        
        # Signal that extraction is complete
        self.extraction_complete.emit(True)
    
    def generate_augmented_templates(self, template, save_dir, sign_type, template_name):
        """Generate augmented variants of the template"""
        # Generate 5 augmented templates
        for i in range(5):
            augmented = template.copy()
            
            # Random rotation +/- 15 degrees
            angle = np.random.uniform(-15, 15)
            rows, cols = augmented.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            augmented = cv2.warpAffine(augmented, M, (cols, rows))
            
            # Random brightness adjustment
            alpha = np.random.uniform(0.7, 1.3)
            beta = np.random.uniform(-30, 30)
            augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)
            
            # Save augmented template
            aug_filename = f"{sign_type}_{template_name}_aug{i}.jpg"
            aug_path = os.path.join(save_dir, aug_filename)
            cv2.imwrite(aug_path, augmented)
    
    def extract_multiple(self):
        """Extract multiple templates (batch mode)"""
        # This would implement a batch extraction interface
        # For now, display a message that it's not implemented
        QMessageBox.information(
            self, "Batch Extraction",
            "Batch template extraction will be implemented.\n"
            "Currently, please extract templates one by one."
        )