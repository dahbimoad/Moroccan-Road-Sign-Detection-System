import os
import cv2
import numpy as np
import shutil
from pathlib import Path

class TemplateManager:
    """
    Utility class for managing sign templates and directory structure
    """
    
    def __init__(self, base_dir=None):
        """
        Initialize the template manager
        
        Args:
            base_dir: Base directory for data, defaults to project's data directory
        """
        # If no base directory is provided, use project's data directory
        if base_dir is None:
            root_dir = Path(__file__).resolve().parent.parent.parent
            self.base_dir = os.path.join(root_dir, "data")
        else:
            self.base_dir = base_dir
            
        # Define paths
        self.templates_dir = os.path.join(self.base_dir, "templates")
        self.regulatory_dir = os.path.join(self.templates_dir, "regulatory")
        self.warning_dir = os.path.join(self.templates_dir, "warning")
        self.information_dir = os.path.join(self.templates_dir, "information")
        self.training_dir = os.path.join(self.base_dir, "training")
        
    def setup_directory_structure(self):
        """
        Create the directory structure for templates and training data
        """
        # Create main directories
        os.makedirs(self.templates_dir, exist_ok=True)
        os.makedirs(self.training_dir, exist_ok=True)
        
        # Create subdirectories for template categories
        os.makedirs(self.regulatory_dir, exist_ok=True)
        os.makedirs(self.warning_dir, exist_ok=True)
        os.makedirs(self.information_dir, exist_ok=True)
        
        # Create subdirectories for specific sign types
        regulatory_signs = ['stop', 'yield', 'no_entry', 'speed_limit_30', 
                           'speed_limit_50', 'speed_limit_60', 'speed_limit_80',
                           'no_parking', 'no_stopping', 'no_overtaking']
                           
        warning_signs = ['warning_general', 'warning_intersection', 'warning_curve',
                        'warning_roundabout', 'warning_pedestrians']
                        
        information_signs = ['information_parking', 'information_hospital', 
                           'information_crosswalk', 'roundabout']
        
        # Create directories for each sign type
        for sign in regulatory_signs:
            os.makedirs(os.path.join(self.regulatory_dir, sign), exist_ok=True)
            
        for sign in warning_signs:
            os.makedirs(os.path.join(self.warning_dir, sign), exist_ok=True)
            
        for sign in information_signs:
            os.makedirs(os.path.join(self.information_dir, sign), exist_ok=True)
            
        print(f"Directory structure created at {self.base_dir}")
        
    def extract_template_from_image(self, image_path, save_path, sign_type, crop_rect=None):
        """
        Extract a template from an image and save it
        
        Args:
            image_path: Path to the image containing the sign
            save_path: Path to save the extracted template
            sign_type: Type of sign (for naming)
            crop_rect: Optional tuple (x, y, w, h) to crop the sign from the image
        
        Returns:
            Path to saved template or None if failed
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None
            
        # If crop rectangle is provided, crop the image
        if crop_rect is not None:
            x, y, w, h = crop_rect
            if (x >= 0 and y >= 0 and
                x + w <= image.shape[1] and
                y + h <= image.shape[0]):
                image = image[y:y+h, x:x+w]
            else:
                print("Error: Crop rectangle is outside image boundaries")
                return None
                
        # Resize to standard size for templates
        resized = cv2.resize(image, (64, 64))
        
        # Create filename
        filename = f"{sign_type}_{os.path.basename(image_path)}"
        save_path_full = os.path.join(save_path, filename)
        
        # Save the template
        cv2.imwrite(save_path_full, resized)
        print(f"Template saved to {save_path_full}")
        
        return save_path_full
        
    def extract_multiple_templates_from_image(self, image_path, save_dir, sign_info_list):
        """
        Extract multiple templates from a single image
        
        Args:
            image_path: Path to the image containing multiple signs
            save_dir: Directory to save the extracted templates
            sign_info_list: List of tuples (sign_type, crop_rect)
            
        Returns:
            List of paths to saved templates
        """
        saved_paths = []
        
        for sign_type, crop_rect in sign_info_list:
            # Determine save location based on sign type
            if sign_type.startswith("speed_limit") or sign_type.startswith("no_"):
                category_dir = self.regulatory_dir
            elif sign_type.startswith("warning"):
                category_dir = self.warning_dir
            else:
                category_dir = self.information_dir
                
            # Ensure the specific sign type directory exists
            sign_dir = os.path.join(category_dir, sign_type)
            os.makedirs(sign_dir, exist_ok=True)
            
            # Extract and save the template
            save_path = self.extract_template_from_image(
                image_path, sign_dir, sign_type, crop_rect
            )
            
            if save_path:
                saved_paths.append(save_path)
                
        return saved_paths
        
    def generate_color_templates(self):
        """
        Generate simple color templates for basic testing
        """
        # Create templates for basic sign shapes and colors
        shapes_dir = os.path.join(self.templates_dir, "shapes")
        os.makedirs(shapes_dir, exist_ok=True)
        
        # Create red circle (prohibition sign)
        red_circle = np.ones((64, 64, 3), dtype=np.uint8) * 255
        cv2.circle(red_circle, (32, 32), 30, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(shapes_dir, "red_circle.png"), red_circle)
        
        # Create red octagon (stop sign)
        red_octagon = np.ones((64, 64, 3), dtype=np.uint8) * 255
        points = []
        for i in range(8):
            angle = i * 45 * np.pi / 180
            x = int(32 + 28 * np.cos(angle))
            y = int(32 + 28 * np.sin(angle))
            points.append([x, y])
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(red_octagon, [pts], (0, 0, 255))
        cv2.imwrite(os.path.join(shapes_dir, "red_octagon.png"), red_octagon)
        
        # Create yellow triangle (warning sign)
        yellow_triangle = np.ones((64, 64, 3), dtype=np.uint8) * 255
        triangle_pts = np.array([[32, 5], [5, 60], [59, 60]], np.int32)
        triangle_pts = triangle_pts.reshape((-1, 1, 2))
        cv2.fillPoly(yellow_triangle, [triangle_pts], (0, 255, 255))
        cv2.imwrite(os.path.join(shapes_dir, "yellow_triangle.png"), yellow_triangle)
        
        # Create blue circle (information sign)
        blue_circle = np.ones((64, 64, 3), dtype=np.uint8) * 255
        cv2.circle(blue_circle, (32, 32), 30, (255, 0, 0), -1)
        cv2.imwrite(os.path.join(shapes_dir, "blue_circle.png"), blue_circle)
        
        print(f"Basic shape templates generated in {shapes_dir}")

def main():
    """
    Set up the template directory structure and generate basic templates
    """
    manager = TemplateManager()
    manager.setup_directory_structure()
    manager.generate_color_templates()
    
    print("\nTemplate manager setup complete.")
    print("You can now extract templates from your own images using:")
    print("  manager = TemplateManager()")
    print("  manager.extract_template_from_image('path/to/image.jpg', 'save/dir', 'sign_type', (x, y, w, h))")

if __name__ == "__main__":
    main()