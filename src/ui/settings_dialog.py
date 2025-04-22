from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                           QComboBox, QCheckBox, QGroupBox, QSpinBox, QDoubleSpinBox,
                           QTabWidget, QWidget, QFormLayout, QSlider, QFileDialog)
from PyQt5.QtCore import Qt, QSettings

class SettingsDialog(QDialog):
    """Dialog to configure application settings"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Settings")
        self.setMinimumWidth(500)
        
        # Load settings
        self.settings = QSettings("MRSD", "MoroccanRoadSignDetection")
        
        # Create layout
        main_layout = QVBoxLayout(self)
        
        # Create tabs
        tabs = QTabWidget()
        
        # Detection tab
        detection_tab = self.create_detection_tab()
        tabs.addTab(detection_tab, "Detection")
        
        # Video tab
        video_tab = self.create_video_tab()
        tabs.addTab(video_tab, "Video")
        
        # Environment tab
        env_tab = self.create_environment_tab()
        tabs.addTab(env_tab, "Environment")
        
        # Output tab
        output_tab = self.create_output_tab()
        tabs.addTab(output_tab, "Output")
        
        # Advanced tab
        advanced_tab = self.create_advanced_tab()
        tabs.addTab(advanced_tab, "Advanced")
        
        main_layout.addWidget(tabs)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # Reset button
        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self.reset_defaults)
        button_layout.addWidget(reset_button)
        
        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        # OK button
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        ok_button.setDefault(True)
        button_layout.addWidget(ok_button)
        
        main_layout.addLayout(button_layout)
    
    def create_detection_tab(self):
        """Create the detection settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Detection threshold
        threshold_group = QGroupBox("Detection Threshold")
        threshold_layout = QFormLayout(threshold_group)
        
        self.threshold_spinner = QDoubleSpinBox()
        self.threshold_spinner.setRange(0.1, 0.9)
        self.threshold_spinner.setSingleStep(0.05)
        self.threshold_spinner.setValue(self.settings.value("detection/threshold", 0.5, float))
        threshold_layout.addRow("Confidence threshold:", self.threshold_spinner)
        
        layout.addWidget(threshold_group)
        
        # Detection mode group
        mode_group = QGroupBox("Detection Mode")
        mode_layout = QFormLayout(mode_group)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Real-time", "Video", "Image", "Batch"])
        current_mode = self.settings.value("detection/mode", "Real-time")
        self.mode_combo.setCurrentText(current_mode)
        mode_layout.addRow("Default mode:", self.mode_combo)
        
        layout.addWidget(mode_group)
        
        # Sign types group
        signs_group = QGroupBox("Sign Types")
        signs_layout = QVBoxLayout(signs_group)
        
        self.regulatory_check = QCheckBox("Regulatory Signs")
        self.regulatory_check.setChecked(self.settings.value("detection/regulatory", True, bool))
        signs_layout.addWidget(self.regulatory_check)
        
        self.warning_check = QCheckBox("Warning Signs")
        self.warning_check.setChecked(self.settings.value("detection/warning", True, bool))
        signs_layout.addWidget(self.warning_check)
        
        self.information_check = QCheckBox("Information Signs")
        self.information_check.setChecked(self.settings.value("detection/information", True, bool))
        signs_layout.addWidget(self.information_check)
        
        layout.addWidget(signs_group)
        
        layout.addStretch()
        return tab
    
    def create_video_tab(self):
        """Create the video settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Camera settings
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QFormLayout(camera_group)
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Default Camera", "External Camera", "IP Camera"])
        current_camera = self.settings.value("video/camera", "Default Camera")
        self.camera_combo.setCurrentText(current_camera)
        camera_layout.addRow("Camera source:", self.camera_combo)
        
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "1280x720", "1920x1080"])
        current_res = self.settings.value("video/resolution", "1280x720")
        self.resolution_combo.setCurrentText(current_res)
        camera_layout.addRow("Resolution:", self.resolution_combo)
        
        self.fps_spinner = QSpinBox()
        self.fps_spinner.setRange(10, 60)
        self.fps_spinner.setValue(self.settings.value("video/fps", 30, int))
        camera_layout.addRow("Target FPS:", self.fps_spinner)
        
        layout.addWidget(camera_group)
        
        # Processing settings
        process_group = QGroupBox("Processing Settings")
        process_layout = QVBoxLayout(process_group)
        
        self.stabilize_check = QCheckBox("Enable video stabilization")
        self.stabilize_check.setChecked(self.settings.value("video/stabilization", False, bool))
        process_layout.addWidget(self.stabilize_check)
        
        self.denoise_check = QCheckBox("Enable denoising")
        self.denoise_check.setChecked(self.settings.value("video/denoising", True, bool))
        process_layout.addWidget(self.denoise_check)
        
        layout.addWidget(process_group)
        
        layout.addStretch()
        return tab
    
    def create_environment_tab(self):
        """Create the environment settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Environment mode
        env_group = QGroupBox("Environment Detection")
        env_layout = QFormLayout(env_group)
        
        self.env_combo = QComboBox()
        self.env_combo.addItems(["Auto", "Urban", "Desert", "Normal"])
        current_env = self.settings.value("environment/type", "Auto")
        self.env_combo.setCurrentText(current_env)
        env_layout.addRow("Environment type:", self.env_combo)
        
        self.auto_detect_check = QCheckBox("Automatically detect environment")
        self.auto_detect_check.setChecked(self.settings.value("environment/autodetect", True, bool))
        env_layout.addRow("", self.auto_detect_check)
        
        layout.addWidget(env_group)
        
        # Lighting adjustments
        lighting_group = QGroupBox("Lighting Adjustments")
        lighting_layout = QFormLayout(lighting_group)
        
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-50, 50)
        self.brightness_slider.setValue(self.settings.value("environment/brightness", 0, int))
        lighting_layout.addRow("Brightness:", self.brightness_slider)
        
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(-50, 50)
        self.contrast_slider.setValue(self.settings.value("environment/contrast", 0, int))
        lighting_layout.addRow("Contrast:", self.contrast_slider)
        
        self.adaptive_check = QCheckBox("Enable adaptive lighting")
        self.adaptive_check.setChecked(self.settings.value("environment/adaptive_lighting", True, bool))
        lighting_layout.addRow("", self.adaptive_check)
        
        layout.addWidget(lighting_group)
        
        layout.addStretch()
        return tab
    
    def create_output_tab(self):
        """Create the output settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Output directory
        dir_group = QGroupBox("Output Directory")
        dir_layout = QHBoxLayout(dir_group)
        
        self.dir_label = QLabel(self.settings.value("output/directory", "output"))
        dir_layout.addWidget(self.dir_label, 1)
        
        self.dir_button = QPushButton("Browse...")
        self.dir_button.clicked.connect(self.browse_directory)
        dir_layout.addWidget(self.dir_button)
        
        layout.addWidget(dir_group)
        
        # Output options
        options_group = QGroupBox("Output Options")
        options_layout = QVBoxLayout(options_group)
        
        self.save_frames_check = QCheckBox("Save processed frames")
        self.save_frames_check.setChecked(self.settings.value("output/save_frames", False, bool))
        options_layout.addWidget(self.save_frames_check)
        
        self.generate_report_check = QCheckBox("Generate detection report")
        self.generate_report_check.setChecked(self.settings.value("output/generate_report", True, bool))
        options_layout.addWidget(self.generate_report_check)
        
        self.extract_templates_check = QCheckBox("Extract detected signs as templates")
        self.extract_templates_check.setChecked(self.settings.value("output/extract_templates", False, bool))
        options_layout.addWidget(self.extract_templates_check)
        
        layout.addWidget(options_group)
        
        # Output format
        format_group = QGroupBox("Output Format")
        format_layout = QFormLayout(format_group)
        
        self.image_format_combo = QComboBox()
        self.image_format_combo.addItems(["JPG", "PNG", "BMP"])
        current_format = self.settings.value("output/image_format", "JPG")
        self.image_format_combo.setCurrentText(current_format)
        format_layout.addRow("Image format:", self.image_format_combo)
        
        self.video_format_combo = QComboBox()
        self.video_format_combo.addItems(["MP4", "AVI", "MKV"])
        current_video_format = self.settings.value("output/video_format", "MP4")
        self.video_format_combo.setCurrentText(current_video_format)
        format_layout.addRow("Video format:", self.video_format_combo)
        
        layout.addWidget(format_group)
        
        layout.addStretch()
        return tab
    
    def create_advanced_tab(self):
        """Create the advanced settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Hardware acceleration
        hardware_group = QGroupBox("Hardware Acceleration")
        hardware_layout = QVBoxLayout(hardware_group)
        
        self.gpu_check = QCheckBox("Use GPU acceleration (if available)")
        self.gpu_check.setChecked(self.settings.value("advanced/use_gpu", False, bool))
        hardware_layout.addWidget(self.gpu_check)
        
        # Add CUDA device selection if checked
        self.cuda_combo = QComboBox()
        self.cuda_combo.addItems(["Default CUDA Device", "CUDA Device 0", "CUDA Device 1"])
        current_device = self.settings.value("advanced/cuda_device", "Default CUDA Device")
        self.cuda_combo.setCurrentText(current_device)
        self.cuda_combo.setEnabled(self.gpu_check.isChecked())
        hardware_layout.addWidget(self.cuda_combo)
        
        # Connect GPU checkbox to enable/disable CUDA device selection
        self.gpu_check.toggled.connect(self.cuda_combo.setEnabled)
        
        layout.addWidget(hardware_group)
        
        # Debug options
        debug_group = QGroupBox("Debug Options")
        debug_layout = QVBoxLayout(debug_group)
        
        self.debug_check = QCheckBox("Enable debug visualization")
        self.debug_check.setChecked(self.settings.value("advanced/debug", False, bool))
        debug_layout.addWidget(self.debug_check)
        
        self.show_steps_check = QCheckBox("Show intermediate processing steps")
        self.show_steps_check.setChecked(self.settings.value("advanced/show_steps", False, bool))
        debug_layout.addWidget(self.show_steps_check)
        
        self.log_check = QCheckBox("Enable detailed logging")
        self.log_check.setChecked(self.settings.value("advanced/logging", False, bool))
        debug_layout.addWidget(self.log_check)
        
        layout.addWidget(debug_group)
        
        # Algorithm settings
        algo_group = QGroupBox("Algorithm Settings")
        algo_layout = QFormLayout(algo_group)
        
        self.temporal_spinner = QDoubleSpinBox()
        self.temporal_spinner.setRange(0.0, 1.0)
        self.temporal_spinner.setSingleStep(0.05)
        self.temporal_spinner.setValue(self.settings.value("advanced/temporal_filter", 0.3, float))
        algo_layout.addRow("Temporal filter threshold:", self.temporal_spinner)
        
        self.nms_spinner = QDoubleSpinBox()
        self.nms_spinner.setRange(0.0, 1.0)
        self.nms_spinner.setSingleStep(0.05)
        self.nms_spinner.setValue(self.settings.value("advanced/nms_threshold", 0.4, float))
        algo_layout.addRow("NMS threshold:", self.nms_spinner)
        
        layout.addWidget(algo_group)
        
        layout.addStretch()
        return tab
    
    def browse_directory(self):
        """Open directory browser dialog"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.dir_label.text())
        
        if directory:
            self.dir_label.setText(directory)
    
    def reset_defaults(self):
        """Reset all settings to default values"""
        # Detection tab
        self.threshold_spinner.setValue(0.5)
        self.mode_combo.setCurrentText("Real-time")
        self.regulatory_check.setChecked(True)
        self.warning_check.setChecked(True)
        self.information_check.setChecked(True)
        
        # Video tab
        self.camera_combo.setCurrentText("Default Camera")
        self.resolution_combo.setCurrentText("1280x720")
        self.fps_spinner.setValue(30)
        self.stabilize_check.setChecked(False)
        self.denoise_check.setChecked(True)
        
        # Environment tab
        self.env_combo.setCurrentText("Auto")
        self.auto_detect_check.setChecked(True)
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)
        self.adaptive_check.setChecked(True)
        
        # Output tab
        self.dir_label.setText("output")
        self.save_frames_check.setChecked(False)
        self.generate_report_check.setChecked(True)
        self.extract_templates_check.setChecked(False)
        self.image_format_combo.setCurrentText("JPG")
        self.video_format_combo.setCurrentText("MP4")
        
        # Advanced tab
        self.gpu_check.setChecked(False)
        self.cuda_combo.setCurrentText("Default CUDA Device")
        self.debug_check.setChecked(False)
        self.show_steps_check.setChecked(False)
        self.log_check.setChecked(False)
        self.temporal_spinner.setValue(0.3)
        self.nms_spinner.setValue(0.4)
    
    def accept(self):
        """Save settings and close dialog"""
        # Detection tab
        self.settings.setValue("detection/threshold", self.threshold_spinner.value())
        self.settings.setValue("detection/mode", self.mode_combo.currentText())
        self.settings.setValue("detection/regulatory", self.regulatory_check.isChecked())
        self.settings.setValue("detection/warning", self.warning_check.isChecked())
        self.settings.setValue("detection/information", self.information_check.isChecked())
        
        # Video tab
        self.settings.setValue("video/camera", self.camera_combo.currentText())
        self.settings.setValue("video/resolution", self.resolution_combo.currentText())
        self.settings.setValue("video/fps", self.fps_spinner.value())
        self.settings.setValue("video/stabilization", self.stabilize_check.isChecked())
        self.settings.setValue("video/denoising", self.denoise_check.isChecked())
        
        # Environment tab
        self.settings.setValue("environment/type", self.env_combo.currentText())
        self.settings.setValue("environment/autodetect", self.auto_detect_check.isChecked())
        self.settings.setValue("environment/brightness", self.brightness_slider.value())
        self.settings.setValue("environment/contrast", self.contrast_slider.value())
        self.settings.setValue("environment/adaptive_lighting", self.adaptive_check.isChecked())
        
        # Output tab
        self.settings.setValue("output/directory", self.dir_label.text())
        self.settings.setValue("output/save_frames", self.save_frames_check.isChecked())
        self.settings.setValue("output/generate_report", self.generate_report_check.isChecked())
        self.settings.setValue("output/extract_templates", self.extract_templates_check.isChecked())
        self.settings.setValue("output/image_format", self.image_format_combo.currentText())
        self.settings.setValue("output/video_format", self.video_format_combo.currentText())
        
        # Advanced tab
        self.settings.setValue("advanced/use_gpu", self.gpu_check.isChecked())
        self.settings.setValue("advanced/cuda_device", self.cuda_combo.currentText())
        self.settings.setValue("advanced/debug", self.debug_check.isChecked())
        self.settings.setValue("advanced/show_steps", self.show_steps_check.isChecked())
        self.settings.setValue("advanced/logging", self.log_check.isChecked())
        self.settings.setValue("advanced/temporal_filter", self.temporal_spinner.value())
        self.settings.setValue("advanced/nms_threshold", self.nms_spinner.value())
        
        self.settings.sync()
        super().accept()