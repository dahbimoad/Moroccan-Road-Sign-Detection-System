import cv2
import os
import argparse
import numpy as np
from pathlib import Path
from template_manager import TemplateManager

class TemplateExtractor:
    """
    Interactive tool to extract templates from images for sign classification
    """
    
    def __init__(self):
        self.template_manager = TemplateManager()
        self.image = None
        self.image_path = None
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.selection_rect = None
        self.sign_type = None
        self.sign_category = None
        
    def setup_window(self):
        """Setup the OpenCV window and callbacks"""
        cv2.namedWindow('Template Extractor')
        cv2.setMouseCallback('Template Extractor', self.mouse_callback)
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse event callback for region selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Create a copy of the original image to draw selection rectangle
                img_copy = self.image.copy()
                cv2.rectangle(img_copy, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                cv2.imshow('Template Extractor', img_copy)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            # Calculate the rectangle coordinates
            x1, y1 = min(self.ix, x), min(self.iy, y)
            x2, y2 = max(self.ix, x), max(self.iy, y)
            w, h = x2 - x1, y2 - y1
            
            # Save the selection rectangle
            self.selection_rect = (x1, y1, w, h)
            
            # Draw the final rectangle
            img_copy = self.image.copy()
            cv2.rectangle(img_copy, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
            cv2.imshow('Template Extractor', img_copy)
    
    def select_sign_category(self):
        """Let the user select the sign category"""
        categories = {
            'r': 'regulatory',
            'w': 'warning',
            'i': 'information'
        }
        
        print("\nSelect sign category:")
        print("  [r] Regulatory (stop, yield, no entry, speed limits)")
        print("  [w] Warning (danger, curves, intersections)")
        print("  [i] Information (directions, services)")
        
        while True:
            key = input("Enter category (r/w/i): ").lower()
            if key in categories:
                self.sign_category = categories[key]
                return
            print("Invalid selection, try again.")
            
    def select_sign_type(self):
        """Let the user select the sign type based on category"""
        
        regulatory_signs = {
            '1': 'stop',
            '2': 'yield',
            '3': 'no_entry',
            '4': 'speed_limit_30',
            '5': 'speed_limit_50',
            '6': 'speed_limit_60',
            '7': 'speed_limit_80',
            '8': 'no_parking',
            '9': 'no_stopping',
            '0': 'no_overtaking'
        }
        
        warning_signs = {
            '1': 'warning_general',
            '2': 'warning_intersection',
            '3': 'warning_curve',
            '4': 'warning_roundabout',
            '5': 'warning_pedestrians'
        }
        
        information_signs = {
            '1': 'information_parking',
            '2': 'information_hospital',
            '3': 'information_crosswalk',
            '4': 'roundabout'
        }
        
        # Choose list based on category
        if self.sign_category == 'regulatory':
            signs = regulatory_signs
            print("\nSelect regulatory sign type:")
            print("  [1] Stop")
            print("  [2] Yield")
            print("  [3] No Entry")
            print("  [4] Speed Limit 30")
            print("  [5] Speed Limit 50")
            print("  [6] Speed Limit 60")
            print("  [7] Speed Limit 80")
            print("  [8] No Parking")
            print("  [9] No Stopping")
            print("  [0] No Overtaking")
            
        elif self.sign_category == 'warning':
            signs = warning_signs
            print("\nSelect warning sign type:")
            print("  [1] General Warning")
            print("  [2] Intersection Warning")
            print("  [3] Curve Warning")
            print("  [4] Roundabout Warning")
            print("  [5] Pedestrian Warning")
            
        else:  # information
            signs = information_signs
            print("\nSelect information sign type:")
            print("  [1] Parking Information")
            print("  [2] Hospital")
            print("  [3] Crosswalk")
            print("  [4] Roundabout")
        
        while True:
            key = input("Enter sign type: ")
            if key in signs:
                self.sign_type = signs[key]
                return
            print("Invalid selection, try again.")
    
    def run(self, image_path):
        """Run the template extractor on an image"""
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        
        if self.image is None:
            print(f"Error: Could not load image from {image_path}")
            return
            
        # Setup window and display image
        self.setup_window()
        cv2.imshow('Template Extractor', self.image)
        print("\nTemplate Extractor")
        print("------------------")
        print("1. Click and drag to select a road sign")
        print("2. Press 's' to save the selection as a template")
        print("3. Press 'c' to clear the current selection")
        print("4. Press 'q' to quit")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') and self.selection_rect is not None:
                # Select sign category and type
                self.select_sign_category()
                self.select_sign_type()
                
                # Get target directory based on category and sign type
                if self.sign_category == 'regulatory':
                    target_dir = os.path.join(self.template_manager.regulatory_dir, self.sign_type)
                elif self.sign_category == 'warning':
                    target_dir = os.path.join(self.template_manager.warning_dir, self.sign_type)
                else:  # information
                    target_dir = os.path.join(self.template_manager.information_dir, self.sign_type)
                
                # Extract and save the template
                self.template_manager.extract_template_from_image(
                    self.image_path, target_dir, self.sign_type, self.selection_rect
                )
                
                print("\nTemplate saved. Select another sign or press 'q' to quit.")
                self.selection_rect = None
                
            elif key == ord('c'):
                # Clear selection
                self.selection_rect = None
                cv2.imshow('Template Extractor', self.image)
                print("\nSelection cleared.")
                
            elif key == ord('q'):
                # Quit
                break
                
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Extract sign templates from images')
    parser.add_argument('image_path', help='Path to the image to extract templates from')
    args = parser.parse_args()
    
    # Ensure the template directory structure exists
    template_manager = TemplateManager()
    template_manager.setup_directory_structure()
    
    # Run the template extractor
    extractor = TemplateExtractor()
    extractor.run(args.image_path)

if __name__ == "__main__":
    main()