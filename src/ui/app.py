import sys
import os
import cv2
import time
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QFileDialog, QComboBox, QTabWidget, 
                           QSlider, QCheckBox, QGroupBox, QGridLayout, QSplitter,
                           QMessageBox, QProgressBar, QScrollArea, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize, QThread

# Add project root to path for imports
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.detection.sign_detector import SignDetector
from src.recognition.sign_classifier import SignClassifier
from src.utils.template_manager import TemplateManager
from src.ui.widgets import SignDetailWidget, ImageViewWidget, FeedbackDialog
from src.ui.about_dialog import AboutDialog
from src.ui.settings_dialog import SettingsDialog
from src.ui.template_management_dialog import TemplateManagementDialog
from src.ui.report_dialog import ReportDialog

class RoadSignApp(QMainWindow):
    """Main application window for Moroccan Road Sign Detection System"""
    
    def __init__(self):
        super().__init__()
        
        # App state
        self.current_image = None
        self.current_frame = None
        self.processed_frame = None
        self.detection_results = []
        self.processing = False
        self.capture = None
        self.camera_running = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_frame)
        
        # Statistics and history
        self.detection_history = []
        self.processing_times = []
        self.detection_counts = {}
        
        # Setup UI elements
        self.setup_ui()
        
        # Initialize system components
        self.preprocessor = ImagePreprocessor()
        self.detector = SignDetector()
        self.classifier = SignClassifier()
        self.template_manager = TemplateManager()
        
        # Set up settings (default values)
        self.settings = {
            "detection_threshold": 0.5,
            "show_debug": False,
            "environment": "auto",
            "extract_templates": False,
            "camera_index": 0,
            "auto_save": False,
            "output_dir": os.path.join(root_dir, "output")
        }
        
        # Ensure output directory exists
        os.makedirs(self.settings["output_dir"], exist_ok=True)
        
        # Connect signals to slots
        self.connect_signals()
    
    def setup_ui(self):
        """Set up the user interface"""
        # Set window properties
        self.setWindowTitle("Moroccan Road Sign Detection System")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(900, 600)
        
        # Set up Moroccan themed colors
        self.setup_theme()
        
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create toolbar layout
        toolbar_layout = QHBoxLayout()
        
        # Create tool buttons
        self.btn_open = QPushButton("Open Image")
        self.btn_open.setIcon(QIcon("assets/icons/open.png"))
        self.btn_open.setFixedHeight(40)
        
        self.btn_camera = QPushButton("Start Camera")
        self.btn_camera.setIcon(QIcon("assets/icons/camera.png"))
        self.btn_camera.setFixedHeight(40)
        
        self.btn_video = QPushButton("Open Video")
        self.btn_video.setIcon(QIcon("assets/icons/video.png"))
        self.btn_video.setFixedHeight(40)
        
        self.env_combo = QComboBox()
        self.env_combo.addItems(["Auto", "Urban", "Desert", "Normal"])
        self.env_combo.setFixedHeight(40)
        self.env_combo.setFixedWidth(120)
        self.env_combo.setCurrentText("Auto")
        
        self.btn_settings = QPushButton("Settings")
        self.btn_settings.setIcon(QIcon("assets/icons/settings.png"))
        self.btn_settings.setFixedHeight(40)
        
        self.btn_about = QPushButton("About")
        self.btn_about.setFixedHeight(40)
        
        # Add buttons to toolbar
        toolbar_layout.addWidget(self.btn_open)
        toolbar_layout.addWidget(self.btn_camera)
        toolbar_layout.addWidget(self.btn_video)
        toolbar_layout.addWidget(QLabel("Environment:"))
        toolbar_layout.addWidget(self.env_combo)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.btn_settings)
        toolbar_layout.addWidget(self.btn_about)
        
        # Add toolbar to main layout
        main_layout.addLayout(toolbar_layout)
        
        # Create splitter for main area and detection panel
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Create main display area (left side)
        main_area = QWidget()
        main_area_layout = QVBoxLayout(main_area)
        
        # Create image view
        self.image_view = ImageViewWidget()
        self.image_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Create controls under image
        controls_layout = QHBoxLayout()
        
        self.btn_process = QPushButton("Process Image")
        self.btn_process.setFixedHeight(40)
        self.btn_process.setEnabled(False)
        
        self.btn_save = QPushButton("Save Result")
        self.btn_save.setFixedHeight(40)
        self.btn_save.setEnabled(False)
        
        self.chk_debug = QCheckBox("Show Debug View")
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(50)  # 0.5 as default
        
        self.lbl_threshold = QLabel("Threshold: 0.50")
        
        controls_layout.addWidget(self.btn_process)
        controls_layout.addWidget(self.btn_save)
        controls_layout.addWidget(self.chk_debug)
        controls_layout.addWidget(QLabel("Detection Threshold:"))
        controls_layout.addWidget(self.threshold_slider)
        controls_layout.addWidget(self.lbl_threshold)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)
        
        # Status bar at bottom of main area
        self.status_label = QLabel("Ready")
        self.fps_label = QLabel("FPS: --")
        status_layout = QHBoxLayout()
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.fps_label)
        
        # Add widgets to main area
        main_area_layout.addWidget(self.image_view)
        main_area_layout.addLayout(controls_layout)
        main_area_layout.addWidget(self.progress_bar)
        main_area_layout.addLayout(status_layout)
        
        # Create detection panel (right side)
        detection_panel = QWidget()
        detection_panel.setMinimumWidth(300)
        detection_panel.setMaximumWidth(400)
        detection_panel_layout = QVBoxLayout(detection_panel)
        
        # Create tabs for detection results and history
        self.tabs = QTabWidget()
        
        # Tab for current detections
        self.current_tab = QWidget()
        current_layout = QVBoxLayout(self.current_tab)
        
        self.detection_scroll = QScrollArea()
        self.detection_scroll.setWidgetResizable(True)
        self.detection_container = QWidget()
        self.detection_layout = QVBoxLayout(self.detection_container)
        self.detection_layout.setAlignment(Qt.AlignTop)
        self.detection_scroll.setWidget(self.detection_container)
        
        current_layout.addWidget(QLabel("<b>Detected Signs:</b>"))
        current_layout.addWidget(self.detection_scroll)
        
        # Tab for history
        self.history_tab = QWidget()
        history_layout = QVBoxLayout(self.history_tab)
        
        self.history_scroll = QScrollArea()
        self.history_scroll.setWidgetResizable(True)
        self.history_container = QWidget()
        self.history_layout = QVBoxLayout(self.history_container)
        self.history_layout.setAlignment(Qt.AlignTop)
        self.history_scroll.setWidget(self.history_container)
        
        self.btn_clear_history = QPushButton("Clear History")
        self.btn_generate_report = QPushButton("Generate Report")
        
        history_buttons = QHBoxLayout()
        history_buttons.addWidget(self.btn_clear_history)
        history_buttons.addWidget(self.btn_generate_report)
        
        history_layout.addWidget(QLabel("<b>Detection History:</b>"))
        history_layout.addWidget(self.history_scroll)
        history_layout.addLayout(history_buttons)
        
        # Tab for statistics
        self.stats_tab = QWidget()
        stats_layout = QVBoxLayout(self.stats_tab)
        
        stats_scroll = QScrollArea()
        stats_scroll.setWidgetResizable(True)
        stats_container = QWidget()
        self.stats_layout = QVBoxLayout(stats_container)
        stats_scroll.setWidget(stats_container)
        
        stats_layout.addWidget(QLabel("<b>Detection Statistics:</b>"))
        stats_layout.addWidget(stats_scroll)
        
        # Add tabs
        self.tabs.addTab(self.current_tab, "Current")
        self.tabs.addTab(self.history_tab, "History")
        self.tabs.addTab(self.stats_tab, "Statistics")
        
        # Add tab widget to detection panel
        detection_panel_layout.addWidget(self.tabs)
        
        # Add template management section
        template_group = QGroupBox("Templates")
        template_layout = QVBoxLayout(template_group)
        
        self.btn_manage_templates = QPushButton("Manage Templates")
        self.btn_extract_templates = QPushButton("Extract from Current")
        self.btn_extract_templates.setEnabled(False)
        
        template_layout.addWidget(self.btn_manage_templates)
        template_layout.addWidget(self.btn_extract_templates)
        
        detection_panel_layout.addWidget(template_group)
        
        # Add main area and detection panel to splitter
        self.splitter.addWidget(main_area)
        self.splitter.addWidget(detection_panel)
        self.splitter.setSizes([700, 300])  # Initial sizes
        
        # Add splitter to main layout
        main_layout.addWidget(self.splitter)
        
        # Set central widget
        self.setCentralWidget(central_widget)
        
        # Initialize empty detection layout display
        self.update_detection_display([])
        self.update_statistics()
    
    def setup_theme(self):
        """Setup Moroccan-inspired color theme for the app"""
        palette = QPalette()
        
        # Moroccan flag green
        moroccan_green = QColor(0, 135, 81)
        # Moroccan flag red
        moroccan_red = QColor(193, 39, 45)
        # Other colors
        light_beige = QColor(245, 240, 230)
        dark_slate = QColor(40, 40, 40)
        
        # Set window background slightly off-white
        palette.setColor(QPalette.Window, light_beige)
        palette.setColor(QPalette.WindowText, dark_slate)
        
        # Button colors
        palette.setColor(QPalette.Button, QColor(220, 220, 220))
        palette.setColor(QPalette.ButtonText, dark_slate)
        
        # Selection colors
        palette.setColor(QPalette.Highlight, moroccan_green)
        palette.setColor(QPalette.HighlightedText, Qt.white)
        
        self.setPalette(palette)
    
    def connect_signals(self):
        """Connect UI signals to slots"""
        # Button connections
        self.btn_open.clicked.connect(self.open_image)
        self.btn_camera.clicked.connect(self.toggle_camera)
        self.btn_video.clicked.connect(self.open_video)
        self.btn_process.clicked.connect(self.process_current_image)
        self.btn_save.clicked.connect(self.save_result)
        self.btn_settings.clicked.connect(self.show_settings)
        self.btn_about.clicked.connect(self.show_about)
        self.btn_manage_templates.clicked.connect(self.show_template_manager)
        self.btn_extract_templates.clicked.connect(self.extract_templates)
        self.btn_clear_history.clicked.connect(self.clear_history)
        self.btn_generate_report.clicked.connect(self.generate_report)
        
        # Slider and checkbox connections
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        self.chk_debug.stateChanged.connect(self.toggle_debug)
        self.env_combo.currentTextChanged.connect(self.change_environment)
    
    def update_threshold_label(self, value):
        """Update threshold label when slider changes"""
        threshold = value / 100.0
        self.settings["detection_threshold"] = threshold
        self.lbl_threshold.setText(f"Threshold: {threshold:.2f}")
    
    def toggle_debug(self, state):
        """Toggle debug mode"""
        self.settings["show_debug"] = (state == Qt.Checked)
        if self.processed_frame is not None:
            self.display_processed_frame()
    
    def change_environment(self, text):
        """Change environment setting"""
        self.settings["environment"] = text.lower()
    
    def open_image(self):
        """Open an image file"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            # Load the image
            self.current_image = cv2.imread(file_path)
            if self.current_image is not None:
                self.current_frame = self.current_image.copy()
                self.processed_frame = None
                self.detection_results = []
                
                # Update UI
                self.display_frame(self.current_frame)
                self.btn_process.setEnabled(True)
                self.status_label.setText(f"Loaded: {os.path.basename(file_path)}")
                
                # Stop camera if running
                if self.camera_running:
                    self.toggle_camera()
            else:
                QMessageBox.critical(self, "Error", "Failed to load the image.")
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if not self.camera_running:
            # Start camera
            self.capture = cv2.VideoCapture(self.settings["camera_index"])
            if not self.capture.isOpened():
                QMessageBox.critical(self, "Error", "Could not open camera")
                return
                
            self.camera_running = True
            self.btn_camera.setText("Stop Camera")
            self.timer.start(30)  # ~33 fps
            self.btn_process.setEnabled(False)
            self.status_label.setText("Camera active")
        else:
            # Stop camera
            if self.capture:
                self.capture.release()
            self.camera_running = False
            self.timer.stop()
            self.btn_camera.setText("Start Camera")
            self.status_label.setText("Camera stopped")
    
    def update_camera_frame(self):
        """Update frame from camera feed"""
        if self.capture is not None:
            ret, frame = self.capture.read()
            if ret:
                # Process frame in real-time
                self.current_frame = frame
                
                # Process frame for detection
                start_time = time.time()
                
                # Process the frame directly
                self.processed_frame = self.process_frame(frame)
                
                # Calculate FPS
                elapsed = time.time() - start_time
                if elapsed > 0:
                    fps = 1.0 / elapsed
                    self.fps_label.setText(f"FPS: {fps:.1f}")
                    self.processing_times.append(elapsed)
                    
                    # Keep only recent times (for moving average)
                    if len(self.processing_times) > 30:
                        self.processing_times.pop(0)
                
                # Display the processed frame
                self.display_frame(self.processed_frame)
                
                # Auto-save if enabled
                if self.settings["auto_save"] and len(self.detection_results) > 0:
                    self.save_frame_with_detection(self.processed_frame)
    
    def open_video(self):
        """Open a video file"""
        # To be implemented - similar to camera but with video file
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if file_path:
            # Stop camera if running
            if self.camera_running:
                self.toggle_camera()
            
            # Show video player dialog
            from .video_player import VideoPlayerDialog
            player = VideoPlayerDialog(self, file_path, self)
            player.exec_()
    
    def process_current_image(self):
        """Process the current still image"""
        if self.current_frame is not None:
            # Disable UI during processing
            self.processing = True
            self.btn_process.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(20)
            
            # Process the image
            self.status_label.setText("Processing image...")
            
            # Create processing thread
            class ProcessThread(QThread):
                finished_signal = pyqtSignal(object, object, list)
                
                def __init__(self, parent, frame):
                    super().__init__(parent)
                    self.frame = frame
                    self.parent = parent
                
                def run(self):
                    start_time = time.time()
                    processed_frame, detection_results = self.parent.process_frame_for_thread(self.frame)
                    elapsed = time.time() - start_time
                    self.finished_signal.emit(processed_frame, elapsed, detection_results)
            
            # Create and start thread
            self.process_thread = ProcessThread(self, self.current_frame)
            self.process_thread.finished_signal.connect(self.on_processing_finished)
            self.process_thread.start()
            
            # Update progress
            self.progress_bar.setValue(50)
        else:
            QMessageBox.warning(self, "Warning", "No image loaded")
    
    def on_processing_finished(self, processed_frame, elapsed, results):
        """Handle processing thread completion"""
        self.processed_frame = processed_frame
        self.detection_results = results
        
        # Update UI
        self.display_processed_frame()
        self.update_detection_display(self.detection_results)
        self.update_history()
        self.update_statistics()
        
        # Update processing time display
        if elapsed > 0:
            fps = 1.0 / elapsed
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.processing_times.append(elapsed)
            
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Processing complete. {len(self.detection_results)} signs detected.")
        self.btn_process.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.btn_extract_templates.setEnabled(len(self.detection_results) > 0)
        self.processing = False
    
    def process_frame(self, frame):
        """Process a single frame and update detection results"""
        processed_frame, detection_results = self.process_frame_for_thread(frame)
        self.detection_results = detection_results
        self.update_detection_display(self.detection_results)
        return processed_frame
    
    def process_frame_for_thread(self, frame):
        """Process a frame for detection (for use in worker thread)"""
        env = self.settings["environment"]
        
        # Apply environment-specific processing
        if env == "auto":
            processed_image = self.preprocessor.detect_by_environment(frame)
        elif env == "urban":
            processed_image = self.preprocessor.enhance_for_urban_conditions(frame)
        elif env == "desert":
            processed_image = self.preprocessor.enhance_for_desert_conditions(frame)
        else:
            processed_image = self.preprocessor.process(frame)
            
        # Detect signs
        detections = self.detector.detect(processed_image)
        
        # Classify and filter by confidence
        results = []
        for x, y, w, h in detections:
            if y >= 0 and y+h <= frame.shape[0] and x >= 0 and x+w <= frame.shape[1]:
                roi = frame[y:y+h, x:x+w]
                
                if roi.size > 0:
                    # Classify the sign
                    sign_type, confidence = self.classifier.classify(roi)
                    
                    # Only include detections above threshold
                    if confidence >= self.settings["detection_threshold"]:
                        # Get sign meaning and category
                        sign_meaning = self.classifier.get_sign_meaning(sign_type)
                        sign_category = self.classifier.get_sign_category(sign_type)
                        
                        results.append({
                            'x': x,
                            'y': y,
                            'w': w,
                            'h': h,
                            'type': sign_type,
                            'category': sign_category,
                            'meaning': sign_meaning,
                            'confidence': confidence,
                            'roi': roi.copy()
                        })
        
        # Draw results
        result_frame = frame.copy()
        result_frame = self.draw_detection_results(result_frame, results)
        
        # Add debug info if enabled
        if self.settings["show_debug"] and results:
            result_frame = self.add_debug_visualization(result_frame, results)
            
        return result_frame, results
    
    def draw_detection_results(self, frame, results):
        """Draw detection boxes on the frame"""
        # Add a semi-transparent overlay for text at the bottom of the frame
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, frame.shape[0] - 40), 
                     (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Add summary information at the bottom
        sign_count = len(results)
        cv2.putText(frame, f"Signs Detected: {sign_count}", 
                   (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        
        # Draw a border around the frame (Moroccan flag colors)
        border_color = (0, 135, 81)  # Green (Moroccan flag color)
        thickness = 3
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), 
                     border_color, thickness)
        
        # Process each detection
        for result in results:
            x, y, w, h = result['x'], result['y'], result['w'], result['h']
            sign_type = result['type']
            confidence = result['confidence']
            category = result['category']
            sign_meaning = result['meaning']
            
            # Determine color based on sign category
            if category == "regulatory":
                box_color = (0, 0, 255)  # Red for regulatory signs
            elif category == "warning":
                box_color = (0, 165, 255)  # Orange for warning signs
            elif category == "information":
                box_color = (255, 0, 0)  # Blue for information signs
            else:
                box_color = (0, 255, 0)  # Green for other signs
            
            # Draw bounding box with more attractive style
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            
            # Add a small colored header above the box
            header_height = 20
            cv2.rectangle(frame, (x, y-header_height), (x+w, y), box_color, -1)
            
            # Add sign type in the header
            cv2.putText(frame, sign_type.replace("_", " ").title(), 
                       (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw label with confidence and sign meaning
            label = f"{sign_meaning} ({confidence:.2f})"
            
            # Create background for better text visibility
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x, y+h), 
                         (x + text_size[0], y+h+text_size[1]+10), (0, 0, 0), -1)
            
            cv2.putText(frame, label, (x, y+h+text_size[1]+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return frame
    
    def add_debug_visualization(self, frame, results):
        """Add debug visualization with Moroccan-styled info panel"""
        # Create a sidebar for debug info
        height = frame.shape[0]
        sidebar_width = 220
        sidebar = np.ones((height, sidebar_width, 3), dtype=np.uint8) * 30  # Dark background
        
        # Add Moroccan flag-inspired design elements
        cv2.rectangle(sidebar, (0, 0), (sidebar_width-1, height-1), (0, 135, 81), 2)  # Green border
        
        # Add header
        cv2.rectangle(sidebar, (0, 0), (sidebar_width, 40), (0, 135, 81), -1)  # Green header
        cv2.putText(sidebar, "Detection Details", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add detected signs to the sidebar
        y_offset = 50
        for idx, data in enumerate(results):
            if idx >= 5:  # Limit to 5 signs to avoid overcrowding
                break
                
            # Add thumbnail
            roi = cv2.resize(data['roi'], (64, 64))
            h, w = roi.shape[:2]
            
            if y_offset + h > height - 50:
                continue
                
            # Add thumbnail border
            cv2.rectangle(sidebar, (10-2, y_offset-2), (10+w+2, y_offset+h+2), (255, 255, 255), 1)
            sidebar[y_offset:y_offset+h, 10:10+w] = roi
            
            # Add category color indicator
            category = data['category']
            if category == "regulatory":
                cat_color = (0, 0, 255)  # Red
                cat_text = "Regulatory"
            elif category == "warning":
                cat_color = (0, 165, 255)  # Orange
                cat_text = "Warning"
            elif category == "information":
                cat_color = (255, 0, 0)  # Blue
                cat_text = "Information"
            else:
                cat_color = (0, 255, 0)  # Green
                cat_text = "Other"
                
            cv2.rectangle(sidebar, (10+w+10, y_offset), (10+w+20, y_offset+h), cat_color, -1)
            
            # Add text details
            cv2.putText(sidebar, f"Type: {data['type']}", (10, y_offset+h+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            cv2.putText(sidebar, f"Conf: {data['confidence']:.2f}", (10, y_offset+h+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            cv2.putText(sidebar, cat_text, (10, y_offset+h+45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, cat_color, 1)
            
            y_offset += h + 60
        
        # Add performance metrics at the bottom
        avg_time = np.mean(self.processing_times[-30:]) if self.processing_times else 0
        fps = 1 / avg_time if avg_time > 0 else 0
        
        cv2.rectangle(sidebar, (0, height-70), (sidebar_width, height), (40, 40, 40), -1)
        cv2.putText(sidebar, f"FPS: {fps:.1f}", (10, height-45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.putText(sidebar, f"Time: {avg_time*1000:.1f}ms", (10, height-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Combine frame and sidebar
        result = np.zeros((height, frame.shape[1] + sidebar_width, 3), dtype=np.uint8)
        result[:, :frame.shape[1]] = frame
        result[:, frame.shape[1]:] = sidebar
        
        return result
    
    def display_frame(self, frame):
        """Display a frame in the image view"""
        if frame is not None:
            self.image_view.update_image(frame)
    
    def display_processed_frame(self):
        """Display the processed frame"""
        if self.processed_frame is not None:
            self.display_frame(self.processed_frame)
            self.btn_save.setEnabled(True)
    
    def update_detection_display(self, detection_results):
        """Update the detection panel with current results"""
        # Clear current display
        for i in reversed(range(self.detection_layout.count())): 
            self.detection_layout.itemAt(i).widget().setParent(None)
        
        if not detection_results:
            no_signs_label = QLabel("No signs detected")
            no_signs_label.setAlignment(Qt.AlignCenter)
            self.detection_layout.addWidget(no_signs_label)
            return
        
        # Add each detection
        for idx, result in detection_results:
            sign_widget = SignDetailWidget(result, idx)
            sign_widget.highlight_signal.connect(self.highlight_detection)
            self.detection_layout.addWidget(sign_widget)
    
    def highlight_detection(self, index):
        """Highlight a specific detection in the image"""
        if self.processed_frame is not None and 0 <= index < len(self.detection_results):
            # Create a copy to avoid modifying the original
            highlight_frame = self.processed_frame.copy()
            result = self.detection_results[index]
            
            # Draw a thicker, flashing border around the selected sign
            x, y, w, h = result['x'], result['y'], result['w'], result['h']
            cv2.rectangle(highlight_frame, (x-5, y-5), (x+w+5, y+h+5), (0, 255, 255), 3)
            
            # Flash the highlight
            self.display_frame(highlight_frame)
            
            # Schedule return to normal display
            QTimer.singleShot(800, self.display_processed_frame)
    
    def update_history(self):
        """Update detection history"""
        if not self.detection_results:
            return
            
        # Add current detections to history
        timestamp = time.time()
        
        for result in self.detection_results:
            history_item = result.copy()
            history_item['timestamp'] = timestamp
            self.detection_history.append(history_item)
        
        # Limit history size
        while len(self.detection_history) > 50:  # Keep last 50 detections
            self.detection_history.pop(0)
        
        # Update history display
        self.update_history_display()
    
    def update_history_display(self):
        """Update the history tab with detection history"""
        # Clear current display
        for i in reversed(range(self.history_layout.count())): 
            self.history_layout.itemAt(i).widget().setParent(None)
        
        if not self.detection_history:
            no_history_label = QLabel("No detection history")
            no_history_label.setAlignment(Qt.AlignCenter)
            self.history_layout.addWidget(no_history_label)
            return
        
        # Sort history by timestamp (newest first)
        sorted_history = sorted(self.detection_history, key=lambda x: x['timestamp'], reverse=True)
        
        # Group by detection time
        grouped_history = {}
        for item in sorted_history:
            timestamp = item['timestamp']
            time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
            
            if time_str not in grouped_history:
                grouped_history[time_str] = []
            grouped_history[time_str].append(item)
        
        # Add each group
        for time_str, items in grouped_history.items():
            # Add time header
            time_label = QLabel(f"<b>{time_str}</b>")
            self.history_layout.addWidget(time_label)
            
            # Add items
            for item in items:
                sign_widget = SignDetailWidget(item, -1, compact=True)
                self.history_layout.addWidget(sign_widget)
            
            # Add separator
            separator = QFrame()
            separator.setFrameShape(QFrame.HLine)
            separator.setFrameShadow(QFrame.Sunken)
            self.history_layout.addWidget(separator)
    
    def update_statistics(self):
        """Update statistics display"""
        # Clear current display
        for i in reversed(range(self.stats_layout.count())): 
            self.stats_layout.itemAt(i).widget().setParent(None)
        
        # Add session stats
        session_group = QGroupBox("Session Statistics")
        session_layout = QGridLayout(session_group)
        
        # Compute statistics
        total_detections = len(self.detection_history)
        sign_types = {}
        sign_categories = {"regulatory": 0, "warning": 0, "information": 0, "other": 0}
        
        for item in self.detection_history:
            sign_type = item['type']
            category = item['category']
            
            if sign_type not in sign_types:
                sign_types[sign_type] = 0
            sign_types[sign_type] += 1
            
            sign_categories[category] += 1
        
        # Add stats in grid
        session_layout.addWidget(QLabel("Total Detections:"), 0, 0)
        session_layout.addWidget(QLabel(str(total_detections)), 0, 1)
        
        session_layout.addWidget(QLabel("Unique Sign Types:"), 1, 0)
        session_layout.addWidget(QLabel(str(len(sign_types))), 1, 1)
        
        # Add processing stats
        avg_time = np.mean(self.processing_times) if self.processing_times else 0
        avg_fps = 1 / avg_time if avg_time > 0 else 0
        
        session_layout.addWidget(QLabel("Avg. Processing Time:"), 2, 0)
        session_layout.addWidget(QLabel(f"{avg_time*1000:.1f} ms"), 2, 1)
        
        session_layout.addWidget(QLabel("Avg. FPS:"), 3, 0)
        session_layout.addWidget(QLabel(f"{avg_fps:.1f}"), 3, 1)
        
        self.stats_layout.addWidget(session_group)
        
        # Add sign type distribution
        if sign_types:
            types_group = QGroupBox("Sign Type Distribution")
            types_layout = QVBoxLayout(types_group)
            
            # Sort by frequency
            sorted_types = sorted(sign_types.items(), key=lambda x: x[1], reverse=True)
            for sign_type, count in sorted_types[:10]:  # Show top 10
                type_layout = QHBoxLayout()
                type_layout.addWidget(QLabel(sign_type.replace("_", " ").title()))
                type_layout.addStretch()
                type_layout.addWidget(QLabel(str(count)))
                types_layout.addLayout(type_layout)
            
            self.stats_layout.addWidget(types_group)
        
        # Add category distribution
        categories_group = QGroupBox("Category Distribution")
        categories_layout = QGridLayout(categories_group)
        
        row = 0
        for category, count in sign_categories.items():
            categories_layout.addWidget(QLabel(category.title()), row, 0)
            categories_layout.addWidget(QLabel(str(count)), row, 1)
            row += 1
        
        self.stats_layout.addWidget(categories_group)
        
        # Add stretch to push everything to the top
        self.stats_layout.addStretch()
    
    def save_result(self):
        """Save the current processed image"""
        if self.processed_frame is None:
            return
            
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "Save Result", "", "Image Files (*.png *.jpg)"
        )
        
        if file_path:
            cv2.imwrite(file_path, self.processed_frame)
            self.status_label.setText(f"Saved result to: {os.path.basename(file_path)}")
    
    def save_frame_with_detection(self, frame):
        """Save a frame automatically when detection occurs"""
        timestamp = int(time.time())
        filename = f"detection_{timestamp}.jpg"
        filepath = os.path.join(self.settings["output_dir"], filename)
        
        cv2.imwrite(filepath, frame)
        print(f"Auto-saved detection to {filepath}")
    
    def extract_templates(self):
        """Extract templates from current detections"""
        if not self.detection_results:
            return
            
        dialog = QMessageBox.question(
            self,
            "Extract Templates",
            f"Extract {len(self.detection_results)} templates from current detections?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if dialog != QMessageBox.Yes:
            return
            
        extracted = 0
        for detection in self.detection_results:
            sign_type = detection['type']
            category = detection['category']
            roi = detection['roi']
            confidence = detection['confidence']
            
            if confidence < 0.7:  # Only use high-confidence detections
                continue
                
            # Determine category directory
            if category == "regulatory":
                category_dir = self.template_manager.regulatory_dir
            elif category == "warning":
                category_dir = self.template_manager.warning_dir
            else:
                category_dir = self.template_manager.information_dir
                
            # Create sign directory
            sign_dir = os.path.join(category_dir, sign_type)
            os.makedirs(sign_dir, exist_ok=True)
            
            # Save template
            timestamp = int(time.time())
            filename = f"{sign_type}_{timestamp}.jpg"
            filepath = os.path.join(sign_dir, filename)
            
            # Resize to standard template size
            template = cv2.resize(roi, (64, 64))
            cv2.imwrite(filepath, template)
            extracted += 1
        
        if extracted > 0:
            QMessageBox.information(
                self,
                "Templates Extracted",
                f"Successfully extracted {extracted} templates."
            )
        else:
            QMessageBox.warning(
                self,
                "No Templates Extracted",
                "No templates were extracted. Try with higher confidence detections."
            )
    
    def show_settings(self):
        """Show settings dialog"""
        dialog = SettingsDialog(self.settings)
        if dialog.exec_():
            # Update settings from dialog
            self.settings = dialog.get_settings()
            
            # Update UI elements to match settings
            self.threshold_slider.setValue(int(self.settings["detection_threshold"] * 100))
            self.chk_debug.setChecked(self.settings["show_debug"])
            self.env_combo.setCurrentText(self.settings["environment"].title())
    
    def show_about(self):
        """Show about dialog"""
        dialog = AboutDialog()
        dialog.exec_()
    
    def show_template_manager(self):
        """Show template management dialog"""
        dialog = TemplateManagementDialog(self.template_manager)
        dialog.exec_()
    
    def clear_history(self):
        """Clear detection history"""
        if not self.detection_history:
            return
            
        dialog = QMessageBox.question(
            self,
            "Clear History",
            "Are you sure you want to clear the detection history?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if dialog == QMessageBox.Yes:
            self.detection_history = []
            self.update_history_display()
            self.update_statistics()
    
    def generate_report(self):
        """Generate a detection report"""
        if not self.detection_history:
            QMessageBox.warning(self, "No Data", "No detection history to generate a report from.")
            return
            
        dialog = ReportDialog(self.detection_history, self.processing_times)
        dialog.exec_()
        
    def closeEvent(self, event):
        """Handle application close event"""
        # Stop camera if running
        if self.camera_running:
            if self.capture:
                self.capture.release()
            self.timer.stop()
        
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = RoadSignApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()