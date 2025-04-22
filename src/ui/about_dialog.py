from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton, 
                           QHBoxLayout, QTabWidget, QWidget, QTextBrowser)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QFont, QIcon

class AboutDialog(QDialog):
    """Dialog to show information about the application"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("About Moroccan Road Sign Detection")
        self.setFixedSize(600, 500)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create header
        header_layout = QVBoxLayout()
        header_layout.setAlignment(Qt.AlignCenter)
        
        # App icon
        icon_label = QLabel()
        icon_label.setPixmap(QIcon(":/icons/app_icon.png").pixmap(QSize(64, 64)))
        icon_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(icon_label)
        
        # App name
        app_name = QLabel("Moroccan Road Sign Detection System")
        app_name.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        app_name.setFont(font)
        header_layout.addWidget(app_name)
        
        # Version
        version = QLabel("Version 1.0.0")
        version.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(version)
        
        # Top container
        top_container = QWidget()
        top_container.setStyleSheet("background-color: #f0f0f0; padding: 20px;")
        top_container.setLayout(header_layout)
        
        layout.addWidget(top_container)
        
        # Create tabs
        tabs = QTabWidget()
        
        # About tab
        about_tab = self.create_about_tab()
        tabs.addTab(about_tab, "About")
        
        # Authors tab
        authors_tab = self.create_authors_tab()
        tabs.addTab(authors_tab, "Authors")
        
        # License tab
        license_tab = self.create_license_tab()
        tabs.addTab(license_tab, "License")
        
        layout.addWidget(tabs)
        
        # Create button layout
        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignCenter)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.setFixedWidth(100)
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
    def create_about_tab(self):
        """Create the about tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        text_browser = QTextBrowser()
        text_browser.setOpenExternalLinks(True)
        
        about_text = """
        <h3>Moroccan Road Sign Detection System</h3>
        
        <p>This application detects and recognizes Moroccan road signs in real-time using computer vision techniques.</p>
        
        <p>Key features:</p>
        <ul>
            <li>Real-time road sign detection</li>
            <li>Support for multiple detection modes (real-time, video, image, batch)</li>
            <li>Environment-specific processing (urban, desert, normal)</li>
            <li>Template-based sign recognition</li>
            <li>Comprehensive reporting and visualization</li>
            <li>GPU acceleration support</li>
        </ul>
        
        <p>Built with Python, OpenCV and PyQt5.</p>
        
        <p>Visit the project website: <a href="https://github.com/yourusername/moroccan-road-sign-detection">https://github.com/yourusername/moroccan-road-sign-detection</a></p>
        """
        
        text_browser.setHtml(about_text)
        layout.addWidget(text_browser)
        
        return tab
    
    def create_authors_tab(self):
        """Create the authors tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        text_browser = QTextBrowser()
        
        authors_text = """
        <h3>Development Team</h3>
        
        <p><b>Project Lead:</b><br>
        Your Name</p>
        
        <p><b>Core Development Team:</b></p>
        <ul>
            <li>Developer 1 - Computer Vision Algorithms</li>
            <li>Developer 2 - User Interface</li>
            <li>Developer 3 - Testing and Validation</li>
        </ul>
        
        <p><b>Acknowledgments:</b></p>
        <p>We would like to thank the Moroccan Ministry of Transport for providing reference materials and sign specifications.</p>
        
        <p>Special thanks to the open source community for the libraries and tools that made this project possible.</p>
        """
        
        text_browser.setHtml(authors_text)
        layout.addWidget(text_browser)
        
        return tab
    
    def create_license_tab(self):
        """Create the license tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        text_browser = QTextBrowser()
        
        license_text = """
        <h3>MIT License</h3>
        
        <p>Copyright (c) 2025 Your Name</p>

        <p>Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:</p>

        <p>The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.</p>

        <p>THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.</p>
        
        <h3>Third-Party Licenses</h3>
        
        <p>This software uses the following third-party open source components:</p>
        
        <ul>
            <li>OpenCV - 3-clause BSD License</li>
            <li>PyQt5 - GPL v3</li>
            <li>NumPy - BSD License</li>
        </ul>
        """
        
        text_browser.setHtml(license_text)
        layout.addWidget(text_browser)
        
        return tab