import os
import shutil
import cv2
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
                           QListWidget, QSplitter, QFileDialog, QMessageBox,
                           QTreeWidget, QTreeWidgetItem, QComboBox, QScrollArea,
                           QFrame, QMenu, QAction, QGridLayout, QWidget)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QColor
from PyQt5.QtCore import Qt, QSize

class TemplateManagementDialog(QDialog):
    """Dialog for managing sign templates"""
    
    def __init__(self, template_manager, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Template Management")
        self.resize(900, 600)
        self.template_manager = template_manager
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create splitter for tree view and template display
        splitter = QSplitter(Qt.Horizontal)
        
        # Create template directory tree (left side)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Templates"])
        self.tree.setMinimumWidth(250)
        self.tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        splitter.addWidget(self.tree)
        
        # Create template display area (right side)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Add filter controls
        filter_layout = QHBoxLayout()
        
        filter_layout.addWidget(QLabel("Filter:"))
        
        self.category_combo = QComboBox()
        self.category_combo.addItems(["All Categories", "Regulatory", "Warning", "Information"])
        filter_layout.addWidget(self.category_combo)
        
        self.search_combo = QComboBox()
        self.search_combo.setEditable(True)
        self.search_combo.setMinimumWidth(150)
        filter_layout.addWidget(self.search_combo)
        
        filter_layout.addStretch()
        
        right_layout.addLayout(filter_layout)
        
        # Add template grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        self.template_container = QWidget()
        self.template_grid = QGridLayout(self.template_container)
        self.template_grid.setAlignment(Qt.AlignTop)
        
        scroll.setWidget(self.template_container)
        right_layout.addWidget(scroll)
        
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes
        splitter.setSizes([250, 650])
        
        layout.addWidget(splitter)
        
        # Add buttons at bottom
        button_layout = QHBoxLayout()
        
        self.import_btn = QPushButton("Import Templates...")
        self.export_btn = QPushButton("Export Selected...")
        self.add_btn = QPushButton("Add from Image...")
        self.delete_btn = QPushButton("Delete Selected")
        self.close_btn = QPushButton("Close")
        
        button_layout.addWidget(self.import_btn)
        button_layout.addWidget(self.export_btn)
        button_layout.addWidget(self.add_btn)
        button_layout.addWidget(self.delete_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        # Connect signals
        self.tree.itemClicked.connect(self.on_tree_selection)
        self.category_combo.currentIndexChanged.connect(self.filter_templates)
        self.search_combo.editTextChanged.connect(self.filter_templates)
        self.search_combo.currentIndexChanged.connect(self.filter_templates)
        
        self.import_btn.clicked.connect(self.import_templates)
        self.export_btn.clicked.connect(self.export_selected)
        self.add_btn.clicked.connect(self.add_from_image)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.close_btn.clicked.connect(self.accept)
        
        # Context menu for tree
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.show_context_menu)
        
        # Load templates
        self.load_template_tree()
        self.populate_search_options()
    
    def load_template_tree(self):
        """Load template directories into tree view"""
        self.tree.clear()
        
        # Create root item for templates
        templates_root = QTreeWidgetItem(self.tree, ["Templates"])
        
        # Add category items
        regulatory_item = QTreeWidgetItem(templates_root, ["Regulatory Signs"])
        warning_item = QTreeWidgetItem(templates_root, ["Warning Signs"])
        info_item = QTreeWidgetItem(templates_root, ["Information Signs"])
        
        # Add regulatory sign types
        self.add_template_directory(regulatory_item, self.template_manager.regulatory_dir)
        
        # Add warning sign types
        self.add_template_directory(warning_item, self.template_manager.warning_dir)
        
        # Add information sign types
        self.add_template_directory(info_item, self.template_manager.information_dir)
        
        # Expand the tree
        self.tree.expandAll()
    
    def add_template_directory(self, parent_item, directory):
        """Add template subdirectories to tree item"""
        if not os.path.exists(directory):
            return
            
        # Get subdirectories
        subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        
        for subdir in sorted(subdirs):
            item = QTreeWidgetItem(parent_item, [subdir.replace("_", " ").title()])
            item.setData(0, Qt.UserRole, os.path.join(directory, subdir))  # Store full path
            
            # Count templates in this directory
            path = os.path.join(directory, subdir)
            count = self.count_templates(path)
            if count > 0:
                item.setText(0, f"{item.text(0)} ({count})")
    
    def count_templates(self, directory):
        """Count template files in directory"""
        if not os.path.exists(directory):
            return 0
            
        return len([f for f in os.listdir(directory) 
                   if os.path.isfile(os.path.join(directory, f)) 
                   and f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    def on_tree_selection(self, item, column):
        """Handle tree item selection"""
        path = item.data(0, Qt.UserRole)
        if path and os.path.isdir(path):
            self.display_templates(path)
    
    def display_templates(self, directory):
        """Display templates from the selected directory"""
        # Clear the template grid
        self.clear_template_grid()
        
        if not os.path.exists(directory):
            return
            
        # Get template files
        files = [f for f in os.listdir(directory) 
                if os.path.isfile(os.path.join(directory, f)) 
                and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not files:
            label = QLabel("No templates in this directory")
            label.setAlignment(Qt.AlignCenter)
            self.template_grid.addWidget(label, 0, 0)
            return
            
        # Add templates to grid
        row, col = 0, 0
        max_cols = 5  # Maximum number of columns
        
        for filename in sorted(files):
            file_path = os.path.join(directory, filename)
            
            # Create template widget
            template_widget = TemplateWidget(file_path)
            self.template_grid.addWidget(template_widget, row, col)
            
            # Update grid position
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
    
    def clear_template_grid(self):
        """Clear all widgets from the template grid"""
        while self.template_grid.count():
            item = self.template_grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    
    def filter_templates(self):
        """Filter templates based on category and search text"""
        category = self.category_combo.currentText()
        search_text = self.search_combo.currentText().lower()
        
        # Implement filtering logic (to be completed)
        # This would require a more comprehensive reorganization of how templates are displayed
        pass
    
    def populate_search_options(self):
        """Populate search dropdown with sign types"""
        # Get all sign types
        sign_types = set()
        
        # Check regulatory directory
        reg_dir = self.template_manager.regulatory_dir
        if os.path.exists(reg_dir):
            sign_types.update([d.replace("_", " ").title() for d in os.listdir(reg_dir) 
                              if os.path.isdir(os.path.join(reg_dir, d))])
        
        # Check warning directory
        warn_dir = self.template_manager.warning_dir
        if os.path.exists(warn_dir):
            sign_types.update([d.replace("_", " ").title() for d in os.listdir(warn_dir) 
                              if os.path.isdir(os.path.join(warn_dir, d))])
        
        # Check information directory
        info_dir = self.template_manager.information_dir
        if os.path.exists(info_dir):
            sign_types.update([d.replace("_", " ").title() for d in os.listdir(info_dir) 
                              if os.path.isdir(os.path.join(info_dir, d))])
        
        # Add items to combo box
        self.search_combo.clear()
        self.search_combo.addItem("")  # Empty item for no filter
        for sign_type in sorted(sign_types):
            self.search_combo.addItem(sign_type)
    
    def import_templates(self):
        """Import templates from a directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Template Directory"
        )
        
        if not dir_path:
            return
            
        # Ask for category
        categories = ["Regulatory", "Warning", "Information"]
        category, ok = QMessageBox.question(
            self, "Select Category",
            "Select the sign category:",
            *categories
        )
        
        if not ok:
            return
            
        # Ask for sign type
        sign_type, ok = QFileDialog.getText(
            self, "Enter Sign Type",
            "Enter the sign type (e.g., stop, yield, speed_limit_50):"
        )
        
        if not ok or not sign_type:
            return
            
        # Determine target directory
        if category == "Regulatory":
            target_dir = os.path.join(self.template_manager.regulatory_dir, sign_type)
        elif category == "Warning":
            target_dir = os.path.join(self.template_manager.warning_dir, sign_type)
        else:
            target_dir = os.path.join(self.template_manager.information_dir, sign_type)
            
        # Create directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy files
        files = [f for f in os.listdir(dir_path) 
                if os.path.isfile(os.path.join(dir_path, f)) 
                and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        count = 0
        for filename in files:
            source_path = os.path.join(dir_path, filename)
            target_path = os.path.join(target_dir, f"{sign_type}_{count}_{filename}")
            
            # Resize to template size
            img = cv2.imread(source_path)
            if img is not None:
                img_resized = cv2.resize(img, (64, 64))
                cv2.imwrite(target_path, img_resized)
                count += 1
        
        # Refresh the view
        self.load_template_tree()
        QMessageBox.information(
            self, "Templates Imported", 
            f"Successfully imported {count} templates."
        )
    
    def export_selected(self):
        """Export selected templates to a directory"""
        # Get selected items
        selected_items = self.tree.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self, "No Selection", 
                "Please select a template category or type to export."
            )
            return
            
        # Get export directory
        export_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Directory"
        )
        
        if not export_dir:
            return
            
        # Export templates from selected items
        exported_count = 0
        for item in selected_items:
            path = item.data(0, Qt.UserRole)
            if path and os.path.isdir(path):
                # Create subdirectory in export directory
                category = os.path.basename(os.path.dirname(path))
                sign_type = os.path.basename(path)
                subdir = os.path.join(export_dir, category, sign_type)
                os.makedirs(subdir, exist_ok=True)
                
                # Copy template files
                files = [f for f in os.listdir(path) 
                        if os.path.isfile(os.path.join(path, f)) 
                        and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                for filename in files:
                    source_path = os.path.join(path, filename)
                    target_path = os.path.join(subdir, filename)
                    shutil.copy2(source_path, target_path)
                    exported_count += 1
        
        QMessageBox.information(
            self, "Templates Exported", 
            f"Successfully exported {exported_count} templates."
        )
    
    def add_from_image(self):
        """Add templates from an image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        
        if not file_path:
            return
            
        # Show template extraction dialog
        from .template_extraction_dialog import TemplateExtractionDialog
        dialog = TemplateExtractionDialog(file_path, self.template_manager, self)
        if dialog.exec_():
            self.load_template_tree()
    
    def delete_selected(self):
        """Delete selected templates or directories"""
        # Get selected items
        selected_items = self.tree.selectedItems()
        if not selected_items:
            return
            
        # Confirm deletion
        message = f"Are you sure you want to delete {len(selected_items)} selected items?"
        if len(selected_items) == 1:
            message = f"Are you sure you want to delete '{selected_items[0].text(0)}'?"
            
        confirm = QMessageBox.question(
            self, "Confirm Deletion", message,
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm != QMessageBox.Yes:
            return
            
        # Delete selected items
        for item in selected_items:
            path = item.data(0, Qt.UserRole)
            if path and os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        
        # Refresh the view
        self.load_template_tree()
    
    def show_context_menu(self, position):
        """Show context menu for tree items"""
        menu = QMenu(self.tree)
        
        # Get selected items
        selected_items = self.tree.selectedItems()
        if not selected_items:
            return
            
        # Add actions
        add_action = QAction("Add Template...", self)
        add_action.triggered.connect(self.add_from_image)
        menu.addAction(add_action)
        
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(self.delete_selected)
        menu.addAction(delete_action)
        
        menu.exec_(self.tree.viewport().mapToGlobal(position))


class TemplateWidget(QFrame):
    """Widget to display a template image with metadata"""
    
    def __init__(self, template_path):
        super().__init__()
        
        # Set frame style
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setLineWidth(1)
        self.setFixedSize(100, 120)
        
        # Store path
        self.template_path = template_path
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)
        
        # Load image
        img = cv2.imread(template_path)
        if img is not None:
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img).scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Create image label
            img_label = QLabel()
            img_label.setPixmap(pixmap)
            img_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(img_label)
            
            # Add file name label
            filename = os.path.basename(template_path)
            if len(filename) > 15:
                filename = filename[:12] + "..."
                
            name_label = QLabel(filename)
            name_label.setAlignment(Qt.AlignCenter)
            name_label.setWordWrap(True)
            layout.addWidget(name_label)
        else:
            # Show error if image couldn't be loaded
            error_label = QLabel("Error")
            error_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(error_label)
    
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.LeftButton:
            # Select the widget
            self.setStyleSheet("background-color: #E0E0FF; border: 2px solid blue;")
        
        super().mousePressEvent(event)
    
    def mouseDoubleClickEvent(self, event):
        """Handle double-click events"""
        if event.button() == Qt.LeftButton:
            # Show preview dialog
            preview = QDialog(self.window())
            preview.setWindowTitle("Template Preview")
            preview_layout = QVBoxLayout(preview)
            
            # Load full image
            img = cv2.imread(self.template_path)
            if img is not None:
                height, width, channel = img.shape
                bytes_per_line = 3 * width
                q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(q_img).scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                # Create image label
                img_label = QLabel()
                img_label.setPixmap(pixmap)
                img_label.setAlignment(Qt.AlignCenter)
                preview_layout.addWidget(img_label)
                
                # Add file info
                filename = os.path.basename(self.template_path)
                file_size = os.path.getsize(self.template_path) / 1024.0  # Size in KB
                
                info_label = QLabel(f"File: {filename}\nSize: {file_size:.1f} KB\nDimensions: {width}x{height}")
                info_label.setAlignment(Qt.AlignCenter)
                preview_layout.addWidget(info_label)
                
                # Add close button
                close_button = QPushButton("Close")
                close_button.clicked.connect(preview.accept)
                preview_layout.addWidget(close_button)
                
                preview.exec_()
        
        super().mouseDoubleClickEvent(event)