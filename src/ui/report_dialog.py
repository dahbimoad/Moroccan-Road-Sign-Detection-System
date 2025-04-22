import os
import time
import json
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, 
                           QPushButton, QLabel, QFileDialog, QComboBox, QFrame,
                           QScrollArea, QGridLayout, QGroupBox, QDialogButtonBox,
                           QMessageBox, QCheckBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2

class ReportDialog(QDialog):
    """Dialog for creating and viewing detection reports"""
    
    def __init__(self, detection_history, processing_times, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Detection Report")
        self.resize(900, 700)
        self.detection_history = detection_history
        self.processing_times = processing_times
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Create tabs
        summary_tab = self.create_summary_tab()
        details_tab = self.create_details_tab()
        charts_tab = self.create_charts_tab()
        export_tab = self.create_export_tab()
        
        self.tabs.addTab(summary_tab, "Summary")
        self.tabs.addTab(details_tab, "Detection Details")
        self.tabs.addTab(charts_tab, "Charts")
        self.tabs.addTab(export_tab, "Export Report")
        
        layout.addWidget(self.tabs)
        
        # Add buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def create_summary_tab(self):
        """Create the summary tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Session summary
        summary_group = QGroupBox("Session Summary")
        summary_layout = QGridLayout(summary_group)
        
        # Calculate summary statistics
        total_signs = len(self.detection_history)
        sign_types = {}
        sign_categories = {"regulatory": 0, "warning": 0, "information": 0, "other": 0}
        
        for item in self.detection_history:
            sign_type = item['type']
            category = item['category']
            
            if sign_type not in sign_types:
                sign_types[sign_type] = 0
            sign_types[sign_type] += 1
            
            sign_categories[category] += 1
        
        avg_time = np.mean(self.processing_times) if self.processing_times else 0
        avg_fps = 1 / avg_time if avg_time > 0 else 0
        
        # Add statistics
        summary_layout.addWidget(QLabel("<b>Total Signs Detected:</b>"), 0, 0)
        summary_layout.addWidget(QLabel(str(total_signs)), 0, 1)
        
        summary_layout.addWidget(QLabel("<b>Unique Sign Types:</b>"), 1, 0)
        summary_layout.addWidget(QLabel(str(len(sign_types))), 1, 1)
        
        summary_layout.addWidget(QLabel("<b>Average Processing Time:</b>"), 2, 0)
        summary_layout.addWidget(QLabel(f"{avg_time*1000:.1f} ms"), 2, 1)
        
        summary_layout.addWidget(QLabel("<b>Average FPS:</b>"), 3, 0)
        summary_layout.addWidget(QLabel(f"{avg_fps:.1f}"), 3, 1)
        
        # Add category breakdown
        summary_layout.addWidget(QLabel("<b>Sign Categories:</b>"), 4, 0, 1, 2)
        
        row = 5
        for category, count in sign_categories.items():
            if count > 0:
                summary_layout.addWidget(QLabel(f"{category.title()}:"), row, 0)
                summary_layout.addWidget(QLabel(str(count)), row, 1)
                row += 1
        
        layout.addWidget(summary_group)
        
        # Top detected signs
        top_signs_group = QGroupBox("Top Detected Signs")
        top_signs_layout = QVBoxLayout(top_signs_group)
        
        # Sort sign types by count
        sorted_types = sorted(sign_types.items(), key=lambda x: x[1], reverse=True)
        
        # Create grid for top signs
        top_grid = QGridLayout()
        
        for i, (sign_type, count) in enumerate(sorted_types[:10]):  # Top 10
            top_grid.addWidget(QLabel(sign_type.replace("_", " ").title()), i, 0)
            top_grid.addWidget(QLabel(str(count)), i, 1)
        
        top_signs_layout.addLayout(top_grid)
        layout.addWidget(top_signs_group)
        
        # Add a visual summary
        visual_group = QGroupBox("Visual Summary")
        visual_layout = QVBoxLayout(visual_group)
        
        # Add a visual representation (pie chart)
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Prepare data for pie chart
        labels = []
        sizes = []
        for category, count in sign_categories.items():
            if count > 0:
                labels.append(category.title())
                sizes.append(count)
        
        if sizes:  # Only create pie chart if there's data
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            ax.set_title("Distribution by Category")
            
            # Add the matplotlib canvas to the layout
            canvas = FigureCanvas(fig)
            visual_layout.addWidget(canvas)
        else:
            visual_layout.addWidget(QLabel("No data available for visualization"))
        
        layout.addWidget(visual_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        return tab
    
    def create_details_tab(self):
        """Create the details tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Filter controls
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter by:"))
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All Signs")
        self.filter_combo.addItem("Regulatory")
        self.filter_combo.addItem("Warning")
        self.filter_combo.addItem("Information")
        filter_layout.addWidget(self.filter_combo)
        
        filter_layout.addStretch()
        layout.addLayout(filter_layout)
        
        # Create scrollable area for detections
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        self.details_container = QWidget()
        self.details_layout = QVBoxLayout(self.details_container)
        self.details_layout.setAlignment(Qt.AlignTop)
        
        scroll.setWidget(self.details_container)
        layout.addWidget(scroll)
        
        # Populate with detection details
        self.populate_detection_details()
        
        # Connect filter combo
        self.filter_combo.currentIndexChanged.connect(self.populate_detection_details)
        
        return tab
    
    def populate_detection_details(self):
        """Populate the details tab with detection information"""
        # Clear existing widgets
        for i in reversed(range(self.details_layout.count())): 
            widget = self.details_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        if not self.detection_history:
            self.details_layout.addWidget(QLabel("No detection history available"))
            return
        
        # Get filter category
        filter_text = self.filter_combo.currentText().lower()
        
        # Group detections by time
        detections_by_time = {}
        for detection in self.detection_history:
            # Apply filtering
            if filter_text != "all signs" and detection['category'] != filter_text:
                continue
                
            # Group by timestamp
            timestamp = detection.get('timestamp', 0)
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            
            if time_str not in detections_by_time:
                detections_by_time[time_str] = []
                
            detections_by_time[time_str].append(detection)
        
        # Check if any detections pass the filter
        if not detections_by_time:
            self.details_layout.addWidget(QLabel(f"No detections found for filter: {filter_text}"))
            return
        
        # Add each group
        for time_str, detections in sorted(detections_by_time.items(), reverse=True):
            # Add time header
            time_label = QLabel(f"<b>{time_str}</b>")
            self.details_layout.addWidget(time_label)
            
            # Add each detection
            for detection in detections:
                # Create detail frame
                detail_frame = QFrame()
                detail_frame.setFrameShape(QFrame.StyledPanel)
                detail_frame.setLineWidth(1)
                
                # Set color based on category
                category = detection['category']
                if category == "regulatory":
                    detail_frame.setStyleSheet("background-color: #FFEDED; border: 1px solid #FF9A9A;")
                elif category == "warning":
                    detail_frame.setStyleSheet("background-color: #FFF7ED; border: 1px solid #FFCC9A;")
                elif category == "information":
                    detail_frame.setStyleSheet("background-color: #EDEFFB; border: 1px solid #9AAAFF;")
                else:
                    detail_frame.setStyleSheet("background-color: #EDFBEF; border: 1px solid #9AFFA7;")
                
                # Create layout
                detail_layout = QHBoxLayout(detail_frame)
                
                # Add thumbnail if ROI is available
                if 'roi' in detection and detection['roi'] is not None:
                    roi = detection['roi']
                    
                    # Convert ROI to QPixmap
                    height, width, channel = roi.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(roi.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                    
                    pixmap = QPixmap.fromImage(q_img).scaled(
                        64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    
                    thumbnail = QLabel()
                    thumbnail.setPixmap(pixmap)
                    thumbnail.setFixedSize(64, 64)
                    detail_layout.addWidget(thumbnail)
                
                # Add text information
                info_layout = QVBoxLayout()
                
                sign_type = detection['type'].replace("_", " ").title()
                type_label = QLabel(f"<b>{sign_type}</b>")
                info_layout.addWidget(type_label)
                
                if 'meaning' in detection:
                    meaning_label = QLabel(detection['meaning'])
                    meaning_label.setWordWrap(True)
                    info_layout.addWidget(meaning_label)
                
                if 'confidence' in detection:
                    conf_label = QLabel(f"Confidence: {detection['confidence']:.2f}")
                    info_layout.addWidget(conf_label)
                
                detail_layout.addLayout(info_layout)
                detail_layout.addStretch()
                
                self.details_layout.addWidget(detail_frame)
            
            # Add separator
            separator = QFrame()
            separator.setFrameShape(QFrame.HLine)
            separator.setFrameShadow(QFrame.Sunken)
            self.details_layout.addWidget(separator)
    
    def create_charts_tab(self):
        """Create the charts tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Check if we have data to visualize
        if not self.detection_history:
            layout.addWidget(QLabel("No detection data available for visualization"))
            return tab
            
        # Chart selection
        chart_layout = QHBoxLayout()
        chart_layout.addWidget(QLabel("Chart Type:"))
        
        self.chart_combo = QComboBox()
        self.chart_combo.addItems([
            "Sign Categories", 
            "Top Sign Types", 
            "Detection Timeline",
            "Confidence Distribution"
        ])
        chart_layout.addWidget(self.chart_combo)
        chart_layout.addStretch()
        
        layout.addLayout(chart_layout)
        
        # Chart container
        self.chart_container = QWidget()
        self.chart_container_layout = QVBoxLayout(self.chart_container)
        
        # Create initial chart
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.chart_container_layout.addWidget(self.canvas)
        
        layout.addWidget(self.chart_container)
        
        # Connect chart selection
        self.chart_combo.currentIndexChanged.connect(self.update_chart)
        
        # Create initial chart
        self.update_chart(0)
        
        return tab
    
    def update_chart(self, index):
        """Update the chart based on selection"""
        # Clear current figure
        self.fig.clear()
        
        # Create new subplot
        ax = self.fig.add_subplot(111)
        
        # Choose chart type based on index
        if index == 0:  # Sign Categories
            self.create_categories_chart(ax)
        elif index == 1:  # Top Sign Types
            self.create_top_types_chart(ax)
        elif index == 2:  # Detection Timeline
            self.create_timeline_chart(ax)
        elif index == 3:  # Confidence Distribution
            self.create_confidence_chart(ax)
        
        # Update the canvas
        self.canvas.draw()
    
    def create_categories_chart(self, ax):
        """Create chart of sign categories"""
        # Count by category
        categories = {"regulatory": 0, "warning": 0, "information": 0, "other": 0}
        
        for detection in self.detection_history:
            category = detection['category']
            categories[category] += 1
            
        # Filter out empty categories
        labels = []
        sizes = []
        colors = ['#FF9A9A', '#FFCC9A', '#9AAAFF', '#9AFFA7']
        filtered_colors = []
        
        for i, (category, count) in enumerate(categories.items()):
            if count > 0:
                labels.append(category.title())
                sizes.append(count)
                filtered_colors.append(colors[i])
        
        # Create pie chart
        if sizes:
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=filtered_colors)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            ax.set_title("Sign Categories")
        else:
            ax.text(0.5, 0.5, "No data available", horizontalalignment='center', verticalalignment='center')
    
    def create_top_types_chart(self, ax):
        """Create chart of top sign types"""
        # Count by sign type
        sign_types = {}
        
        for detection in self.detection_history:
            sign_type = detection['type']
            
            if sign_type not in sign_types:
                sign_types[sign_type] = 0
                
            sign_types[sign_type] += 1
        
        # Sort by count and take top 10
        sorted_types = sorted(sign_types.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if sorted_types:
            # Prepare data
            types = [t[0].replace("_", " ").title() for t in sorted_types]
            counts = [t[1] for t in sorted_types]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(types))
            ax.barh(y_pos, counts, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(types)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Count')
            ax.set_title('Top Sign Types')
            
            # Add count labels
            for i, v in enumerate(counts):
                ax.text(v + 0.1, i, str(v), va='center')
        else:
            ax.text(0.5, 0.5, "No data available", horizontalalignment='center', verticalalignment='center')
    
    def create_timeline_chart(self, ax):
        """Create detection timeline chart"""
        # Group detections by timestamp
        times = {}
        
        for detection in self.detection_history:
            timestamp = detection.get('timestamp', 0)
            time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
            
            if time_str not in times:
                times[time_str] = 0
                
            times[time_str] += 1
        
        if times:
            # Prepare data
            times_sorted = sorted(times.items())
            time_labels = [t[0] for t in times_sorted]
            counts = [t[1] for t in times_sorted]
            
            # Create bar chart
            ax.bar(time_labels, counts)
            ax.set_ylabel('Number of Detections')
            ax.set_xlabel('Time')
            ax.set_title('Detection Timeline')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Make sure labels fit
            self.fig.tight_layout()
        else:
            ax.text(0.5, 0.5, "No timeline data available", horizontalalignment='center', verticalalignment='center')
    
    def create_confidence_chart(self, ax):
        """Create confidence distribution chart"""
        # Extract confidence values
        confidences = [detection['confidence'] for detection in self.detection_history if 'confidence' in detection]
        
        if confidences:
            # Create histogram
            ax.hist(confidences, bins=10, range=(0, 1), edgecolor='black')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Number of Detections')
            ax.set_title('Confidence Distribution')
            
            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
        else:
            ax.text(0.5, 0.5, "No confidence data available", horizontalalignment='center', verticalalignment='center')
    
    def create_export_tab(self):
        """Create the export tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add export options
        options_group = QGroupBox("Export Options")
        options_layout = QVBoxLayout(options_group)
        
        # Format selection
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Report Format:"))
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PDF", "HTML", "JSON"])
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        
        options_layout.addLayout(format_layout)
        
        # Content options
        self.include_images_check = QCheckBox("Include detection images")
        self.include_images_check.setChecked(True)
        options_layout.addWidget(self.include_images_check)
        
        self.include_charts_check = QCheckBox("Include charts and graphs")
        self.include_charts_check.setChecked(True)
        options_layout.addWidget(self.include_charts_check)
        
        self.include_details_check = QCheckBox("Include detailed detection information")
        self.include_details_check.setChecked(True)
        options_layout.addWidget(self.include_details_check)
        
        layout.addWidget(options_group)
        
        # Export destination
        destination_group = QGroupBox("Export Destination")
        destination_layout = QHBoxLayout(destination_group)
        
        self.export_path = QLabel("/path/to/export/destination")
        destination_layout.addWidget(self.export_path)
        
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_export_path)
        destination_layout.addWidget(browse_button)
        
        layout.addWidget(destination_group)
        
        # Export button
        export_button = QPushButton("Generate Report")
        export_button.clicked.connect(self.export_report)
        layout.addWidget(export_button)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        return tab
    
    def browse_export_path(self):
        """Browse for export destination"""
        # Determine file extension based on format
        format_type = self.format_combo.currentText().lower()
        file_filter = ""
        
        if format_type == "pdf":
            file_filter = "PDF Files (*.pdf)"
        elif format_type == "html":
            file_filter = "HTML Files (*.html)"
        elif format_type == "json":
            file_filter = "JSON Files (*.json)"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", "", file_filter
        )
        
        if file_path:
            self.export_path.setText(file_path)
    
    def export_report(self):
        """Export the report in the selected format"""
        export_path = self.export_path.text()
        format_type = self.format_combo.currentText().lower()
        
        # Validate path
        if export_path == "/path/to/export/destination":
            QMessageBox.warning(self, "Export Error", "Please select a valid export destination.")
            return
        
        # Create report data
        report_data = self.generate_report_data()
        
        try:
            # Export in the selected format
            if format_type == "pdf":
                self.export_pdf(export_path, report_data)
            elif format_type == "html":
                self.export_html(export_path, report_data)
            elif format_type == "json":
                self.export_json(export_path, report_data)
                
            QMessageBox.information(self, "Report Generated", f"Report successfully exported to {export_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"An error occurred while exporting: {str(e)}")
    
    def generate_report_data(self):
        """Generate the report data"""
        # Gather statistics
        total_signs = len(self.detection_history)
        sign_types = {}
        sign_categories = {"regulatory": 0, "warning": 0, "information": 0, "other": 0}
        
        for item in self.detection_history:
            sign_type = item['type']
            category = item['category']
            
            if sign_type not in sign_types:
                sign_types[sign_type] = 0
            sign_types[sign_type] += 1
            
            sign_categories[category] += 1
        
        avg_time = np.mean(self.processing_times) if self.processing_times else 0
        avg_fps = 1 / avg_time if avg_time > 0 else 0
        
        # Create report data structure
        report = {
            "summary": {
                "total_signs": total_signs,
                "unique_types": len(sign_types),
                "categories": sign_categories,
                "avg_processing_time_ms": avg_time * 1000,
                "avg_fps": avg_fps
            },
            "sign_types": sign_types,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Include detection details if selected
        if self.include_details_check.isChecked():
            # Convert detections to serializable format
            detections = []
            for detection in self.detection_history:
                serializable = {
                    "type": detection['type'],
                    "category": detection['category'],
                    "meaning": detection.get('meaning', ""),
                    "confidence": detection.get('confidence', 0),
                    "timestamp": detection.get('timestamp', 0)
                }
                
                # Only include ROI if images are selected
                if self.include_images_check.isChecked() and 'roi' in detection:
                    # We'll handle images separately for each format
                    pass
                
                detections.append(serializable)
                
            report["detections"] = detections
        
        return report
    
    def export_pdf(self, path, report_data):
        """Export report as PDF"""
        # PDF export would typically use a library like ReportLab
        # This is a placeholder for the actual implementation
        QMessageBox.information(
            self, "PDF Export",
            "PDF export requires the ReportLab library.\n"
            "For now, consider exporting to HTML or JSON."
        )
    
    def export_html(self, path, report_data):
        """Export report as HTML"""
        # Basic HTML report
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Road Sign Detection Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { text-align: center; margin-bottom: 30px; }
                .section { margin-bottom: 30px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .regulatory { background-color: #FFEDED; }
                .warning { background-color: #FFF7ED; }
                .information { background-color: #EDEFFB; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Road Sign Detection Report</h1>
                <p>Generated on: {timestamp}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <table>
                    <tr><th>Total Signs Detected</th><td>{total_signs}</td></tr>
                    <tr><th>Unique Sign Types</th><td>{unique_types}</td></tr>
                    <tr><th>Average Processing Time</th><td>{avg_time:.1f} ms</td></tr>
                    <tr><th>Average FPS</th><td>{avg_fps:.1f}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Category Distribution</h2>
                <table>
                    <tr><th>Category</th><th>Count</th></tr>
                    {category_rows}
                </table>
            </div>
            
            <div class="section">
                <h2>Top Sign Types</h2>
                <table>
                    <tr><th>Type</th><th>Count</th></tr>
                    {type_rows}
                </table>
            </div>
        """.format(
            timestamp=report_data["timestamp"],
            total_signs=report_data["summary"]["total_signs"],
            unique_types=report_data["summary"]["unique_types"],
            avg_time=report_data["summary"]["avg_processing_time_ms"],
            avg_fps=report_data["summary"]["avg_fps"],
            category_rows="\n".join([f"<tr><td>{cat.title()}</td><td>{count}</td></tr>" 
                                    for cat, count in report_data["summary"]["categories"].items() 
                                    if count > 0]),
            type_rows="\n".join([f"<tr><td>{sign_type.replace('_', ' ').title()}</td><td>{count}</td></tr>" 
                               for sign_type, count in sorted(report_data["sign_types"].items(), 
                                                             key=lambda x: x[1], reverse=True)[:10]])
        )
        
        # Add detection details if included
        if self.include_details_check.isChecked() and "detections" in report_data:
            detection_rows = []
            for detection in report_data["detections"]:
                time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(detection.get("timestamp", 0)))
                row = f"""
                <tr class="{detection['category']}">
                    <td>{detection['type'].replace('_', ' ').title()}</td>
                    <td>{detection['category'].title()}</td>
                    <td>{detection['meaning']}</td>
                    <td>{detection['confidence']:.2f}</td>
                    <td>{time_str}</td>
                </tr>
                """
                detection_rows.append(row)
            
            html += """
            <div class="section">
                <h2>Detection Details</h2>
                <table>
                    <tr>
                        <th>Type</th>
                        <th>Category</th>
                        <th>Meaning</th>
                        <th>Confidence</th>
                        <th>Time</th>
                    </tr>
                    {detection_rows}
                </table>
            </div>
            """.format(detection_rows="\n".join(detection_rows))
        
        # Close HTML
        html += """
        </body>
        </html>
        """
        
        # Write to file
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
    
    def export_json(self, path, report_data):
        """Export report as JSON"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)