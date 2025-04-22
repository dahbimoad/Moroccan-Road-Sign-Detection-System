import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, 
                           QFrame, QDialog, QTextEdit, QFormLayout, QDialogButtonBox)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QSize

class SignDetailWidget(QFrame):
    """Widget to display details about a detected road sign"""
    
    highlight_signal = pyqtSignal(int)
    
    def __init__(self, sign_data, idx, compact=False):
        super().__init__()
        
        self.sign_data = sign_data
        self.index = idx
        
        # Set frame style
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setLineWidth(2)
        
        # Set background color based on sign category
        category = sign_data['category']
        
        if category == "regulatory":
            self.setStyleSheet("background-color: #FFEDED; border: 1px solid #FF9A9A;")
        elif category == "warning":
            self.setStyleSheet("background-color: #FFF7ED; border: 1px solid #FFCC9A;")
        elif category == "information":
            self.setStyleSheet("background-color: #EDEFFB; border: 1px solid #9AAAFF;")
        else:
            self.setStyleSheet("background-color: #EDFBEF; border: 1px solid #9AFFA7;")
        
        # Create layout
        layout = QHBoxLayout(self)
        
        # Add thumbnail if ROI is available
        if 'roi' in sign_data and sign_data['roi'] is not None:
            # Convert ROI to QPixmap
            roi = sign_data['roi']
            height, width, channel = roi.shape
            bytes_per_line = 3 * width
            q_img = QImage(roi.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            # Create thumbnail label
            thumbnail_size = 48 if not compact else 32
            pixmap = QPixmap.fromImage(q_img).scaled(
                thumbnail_size, thumbnail_size, 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            thumbnail = QLabel()
            thumbnail.setPixmap(pixmap)
            thumbnail.setFixedSize(thumbnail_size, thumbnail_size)
            layout.addWidget(thumbnail)
        
        # Create info widget
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add sign type
        sign_type = sign_data['type'].replace("_", " ").title()
        type_label = QLabel(f"<b>{sign_type}</b>")
        type_label.setFont(QFont("Arial", 10))
        info_layout.addWidget(type_label)
        
        # Add confidence and meaning (truncated if compact)
        if 'meaning' in sign_data:
            meaning = sign_data['meaning']
            if compact and len(meaning) > 30:
                meaning = meaning[:27] + "..."
                
            meaning_label = QLabel(meaning)
            meaning_label.setWordWrap(not compact)
            info_layout.addWidget(meaning_label)
            
        if 'confidence' in sign_data:
            confidence = sign_data['confidence']
            conf_label = QLabel(f"Confidence: {confidence:.2f}")
            conf_label.setFont(QFont("Arial", 8))
            info_layout.addWidget(conf_label)
            
        layout.addWidget(info_widget)
        
        # Add highlight button if not in compact mode and has a valid index
        if not compact and idx >= 0:
            highlight_btn = QPushButton("🔍")
            highlight_btn.setFixedSize(30, 30)
            highlight_btn.setToolTip("Highlight in image")
            highlight_btn.clicked.connect(self.highlight_sign)
            
            layout.addWidget(highlight_btn)
            
        # Set max height for compact mode
        if compact:
            self.setMaximumHeight(70)
            
    def highlight_sign(self):
        """Emit signal to highlight this sign in the main image"""
        self.highlight_signal.emit(self.index)


class ImageViewWidget(QLabel):
    """Widget to display and interact with images"""
    
    def __init__(self):
        super().__init__()
        
        # Set placeholder background
        self.setText("No image loaded")
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #F0F0F0;")
        
        # Image data
        self.image_data = None
        self.scale_factor = 1.0
        
    def update_image(self, image):
        """Update the displayed image"""
        if image is None:
            return
            
        # Store original image
        self.image_data = image.copy() if isinstance(image, np.ndarray) else None
        
        # Convert OpenCV image to Qt format
        if isinstance(image, np.ndarray):
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            
            # Set pixmap
            self.setPixmap(pixmap)
            
            # Reset scaling
            self.scale_factor = 1.0
        else:
            # If not an image, show error
            self.setText("Invalid image format")
            self.setPixmap(QPixmap())
            
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming"""
        if self.image_data is not None:
            zoom_in = event.angleDelta().y() > 0
            
            # Change scale factor
            if zoom_in:
                self.scale_factor *= 1.1
                if self.scale_factor > 5.0:
                    self.scale_factor = 5.0
            else:
                self.scale_factor /= 1.1
                if self.scale_factor < 0.1:
                    self.scale_factor = 0.1
            
            # Get original dimensions
            height, width = self.image_data.shape[:2]
            
            # Calculate new dimensions
            new_width = int(width * self.scale_factor)
            new_height = int(height * self.scale_factor)
            
            # Resize image
            resized = cv2.resize(self.image_data, (new_width, new_height))
            
            # Convert to QPixmap and display
            bytes_per_line = 3 * new_width
            q_img = QImage(resized.data, new_width, new_height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            self.setPixmap(pixmap)
            
            # Update tooltip with zoom level
            self.setToolTip(f"Zoom: {self.scale_factor:.1f}x")
            
            
class FeedbackDialog(QDialog):
    """Dialog for user feedback on detection quality"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Detection Feedback")
        self.setMinimumWidth(400)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Add instructions
        layout.addWidget(QLabel("Please provide feedback on the detection quality:"))
        
        # Form layout for ratings
        form = QFormLayout()
        
        # Add ratings inputs here (to be implemented)
        
        layout.addLayout(form)
        
        # Add comments field
        layout.addWidget(QLabel("Additional Comments:"))
        self.comments_field = QTextEdit()
        layout.addWidget(self.comments_field)
        
        # Add buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def get_feedback(self):
        """Get feedback data from the dialog"""
        return {
            "comments": self.comments_field.toPlainText()
            # Add additional feedback fields here
        }