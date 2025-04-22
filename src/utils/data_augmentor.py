import cv2
import os
import numpy as np
import random
from pathlib import Path
from template_manager import TemplateManager

class DataAugmentor:
    """
    Generate synthetic training data through various augmentation techniques
    to improve the robustness of sign detection
    """
    
    def __init__(self, template_manager=None):
        """
        Initialize the data augmentor
        
        Args:
            template_manager: TemplateManager instance, created if not provided
        """
        self.template_manager = template_manager or TemplateManager()
        
    def augment_template(self, template_path, output_dir, num_augmentations=10):
        """
        Apply various augmentations to a template image
        
        Args:
            template_path: Path to the template image
            output_dir: Directory to save augmented images
            num_augmentations: Number of augmentations to generate
            
        Returns:
            List of paths to augmented images
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load template image
        template = cv2.imread(template_path)
        if template is None:
            print(f"Error: Could not read template image {template_path}")
            return []
            
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(template_path))[0]
        
        augmented_paths = []
        
        for i in range(num_augmentations):
            # Apply random augmentations
            augmented = template.copy()
            
            # 1. Random rotation (up to 15 degrees)
            angle = random.uniform(-15, 15)
            h, w = augmented.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented = cv2.warpAffine(augmented, M, (w, h), 
                                      borderMode=cv2.BORDER_REFLECT)
            
            # 2. Random brightness and contrast adjustment
            alpha = random.uniform(0.8, 1.2)  # Contrast
            beta = random.uniform(-20, 20)    # Brightness
            augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)
            
            # 3. Random noise
            if random.random() < 0.5:
                noise = np.random.normal(0, 10, augmented.shape).astype(np.uint8)
                augmented = cv2.add(augmented, noise)
            
            # 4. Random blur
            if random.random() < 0.3:
                kernel_size = random.choice([3, 5])
                augmented = cv2.GaussianBlur(augmented, (kernel_size, kernel_size), 0)
                
            # 5. Random perspective transform (subtle)
            if random.random() < 0.4:
                h, w = augmented.shape[:2]
                # Define mild perspective changes
                offset = w * 0.05  # 5% of width as max offset
                src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
                dst_points = np.float32([
                    [random.uniform(0, offset), random.uniform(0, offset)],
                    [w - random.uniform(0, offset), random.uniform(0, offset)],
                    [w - random.uniform(0, offset), h - random.uniform(0, offset)],
                    [random.uniform(0, offset), h - random.uniform(0, offset)]
                ])
                
                # Apply perspective transform
                M = cv2.getPerspectiveTransform(src_points, dst_points)
                augmented = cv2.warpPerspective(augmented, M, (w, h))
            
            # Save the augmented image
            output_path = os.path.join(output_dir, f"{base_name}_aug_{i+1}.jpg")
            cv2.imwrite(output_path, augmented)
            augmented_paths.append(output_path)
            
        return augmented_paths
        
    def augment_all_templates(self, category=None, sign_type=None, num_per_template=10):
        """
        Augment all templates in the templates directory
        
        Args:
            category: Optional category to filter templates (regulatory, warning, information)
            sign_type: Optional sign type to filter templates
            num_per_template: Number of augmentations to generate per template
            
        Returns:
            Total number of augmented images generated
        """
        # Set up directory paths
        templates_dir = self.template_manager.templates_dir
        training_dir = self.template_manager.training_dir
        
        # Create training directory structure mirroring templates
        if category:
            categories = [category]
        else:
            categories = ['regulatory', 'warning', 'information']
            
        total_augmented = 0
        
        # Process each category
        for cat in categories:
            cat_dir = os.path.join(templates_dir, cat)
            if not os.path.exists(cat_dir) or not os.path.isdir(cat_dir):
                continue
                
            # Create corresponding training directory
            train_cat_dir = os.path.join(training_dir, cat)
            os.makedirs(train_cat_dir, exist_ok=True)
            
            # Process each sign type directory
            sign_types = [sign_type] if sign_type else os.listdir(cat_dir)
            for st in sign_types:
                sign_dir = os.path.join(cat_dir, st)
                if not os.path.exists(sign_dir) or not os.path.isdir(sign_dir):
                    continue
                    
                # Create corresponding training directory for sign type
                train_sign_dir = os.path.join(train_cat_dir, st)
                os.makedirs(train_sign_dir, exist_ok=True)
                
                # Process each template in the sign directory
                for filename in os.listdir(sign_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        template_path = os.path.join(sign_dir, filename)
                        
                        # Augment this template
                        augmented_paths = self.augment_template(
                            template_path, train_sign_dir, num_per_template
                        )
                        
                        total_augmented += len(augmented_paths)
                        print(f"Generated {len(augmented_paths)} augmented images for {template_path}")
        
        return total_augmented
        
    def generate_synthetic_backgrounds(self, num_backgrounds=10, size=(640, 480)):
        """
        Generate synthetic backgrounds for testing sign detection
        
        Args:
            num_backgrounds: Number of backgrounds to generate
            size: Size of backgrounds (width, height)
            
        Returns:
            List of paths to generated background images
        """
        output_dir = os.path.join(self.template_manager.base_dir, "backgrounds")
        os.makedirs(output_dir, exist_ok=True)
        
        background_paths = []
        
        for i in range(num_backgrounds):
            # Create a synthetic background with gradients and random patterns
            bg_type = random.choice(['gradient', 'noise', 'pattern'])
            
            if bg_type == 'gradient':
                # Create a gradient background
                bg = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                
                # Random gradient direction and colors
                start_color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )
                
                end_color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )
                
                # Horizontal or vertical gradient
                if random.random() < 0.5:
                    # Horizontal gradient
                    for x in range(size[0]):
                        ratio = x / size[0]
                        color = (
                            int(start_color[0] * (1 - ratio) + end_color[0] * ratio),
                            int(start_color[1] * (1 - ratio) + end_color[1] * ratio),
                            int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
                        )
                        cv2.line(bg, (x, 0), (x, size[1]), color, 1)
                else:
                    # Vertical gradient
                    for y in range(size[1]):
                        ratio = y / size[1]
                        color = (
                            int(start_color[0] * (1 - ratio) + end_color[0] * ratio),
                            int(start_color[1] * (1 - ratio) + end_color[1] * ratio),
                            int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
                        )
                        cv2.line(bg, (0, y), (size[0], y), color, 1)
            
            elif bg_type == 'noise':
                # Create a noise background
                bg = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
                # Smooth the noise
                bg = cv2.GaussianBlur(bg, (21, 21), 0)
                
            else:  # pattern
                # Create a pattern background
                bg = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                
                # Add some random shapes
                for _ in range(random.randint(5, 15)):
                    shape_type = random.choice(['circle', 'rectangle', 'line'])
                    color = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )
                    
                    if shape_type == 'circle':
                        center = (
                            random.randint(0, size[0]),
                            random.randint(0, size[1])
                        )
                        radius = random.randint(20, 100)
                        cv2.circle(bg, center, radius, color, -1)
                        
                    elif shape_type == 'rectangle':
                        x1 = random.randint(0, size[0])
                        y1 = random.randint(0, size[1])
                        x2 = random.randint(x1, size[0])
                        y2 = random.randint(y1, size[1])
                        cv2.rectangle(bg, (x1, y1), (x2, y2), color, -1)
                        
                    else:  # line
                        x1 = random.randint(0, size[0])
                        y1 = random.randint(0, size[1])
                        x2 = random.randint(0, size[0])
                        y2 = random.randint(0, size[1])
                        thickness = random.randint(1, 10)
                        cv2.line(bg, (x1, y1), (x2, y2), color, thickness)
            
            # Apply blur to make it more natural
            bg = cv2.GaussianBlur(bg, (5, 5), 0)
            
            # Save the background
            output_path = os.path.join(output_dir, f"background_{i+1}.jpg")
            cv2.imwrite(output_path, bg)
            background_paths.append(output_path)
            
        return background_paths
        
    def composite_signs_on_backgrounds(self, num_composites=20):
        """
        Create composite images by placing sign templates on backgrounds
        
        Args:
            num_composites: Number of composite images to generate
            
        Returns:
            List of paths to generated composite images
        """
        output_dir = os.path.join(self.template_manager.base_dir, "composites")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get backgrounds or generate if none exist
        bg_dir = os.path.join(self.template_manager.base_dir, "backgrounds")
        if not os.path.exists(bg_dir) or len(os.listdir(bg_dir)) == 0:
            self.generate_synthetic_backgrounds(10)
            
        bg_files = [f for f in os.listdir(bg_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Get all template files
        templates = []
        categories = ['regulatory', 'warning', 'information']
        
        for cat in categories:
            cat_dir = os.path.join(self.template_manager.templates_dir, cat)
            if not os.path.exists(cat_dir) or not os.path.isdir(cat_dir):
                continue
                
            for sign_dir in os.listdir(cat_dir):
                sign_path = os.path.join(cat_dir, sign_dir)
                if not os.path.isdir(sign_path):
                    continue
                    
                for filename in os.listdir(sign_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        template_path = os.path.join(sign_path, filename)
                        templates.append((template_path, sign_dir, cat))
        
        if not templates:
            print("No templates found. Please add templates first.")
            return []
            
        if not bg_files:
            print("No backgrounds found. Please generate backgrounds first.")
            return []
        
        composite_paths = []
        
        for i in range(num_composites):
            # Select a random background
            bg_file = random.choice(bg_files)
            bg_path = os.path.join(bg_dir, bg_file)
            bg = cv2.imread(bg_path)
            
            if bg is None:
                continue
                
            bg_height, bg_width = bg.shape[:2]
            
            # Decide how many signs to place (1-3)
            num_signs = random.randint(1, 3)
            
            # Keep track of sign positions to avoid overlap
            positions = []
            
            for j in range(num_signs):
                # Select a random template
                template_path, sign_type, category = random.choice(templates)
                template = cv2.imread(template_path)
                
                if template is None:
                    continue
                    
                # Resize template (randomly between 5-15% of background size)
                size_factor = random.uniform(0.05, 0.15)
                new_width = int(bg_width * size_factor)
                new_height = int(template.shape[0] * new_width / template.shape[1])
                resized = cv2.resize(template, (new_width, new_height))
                
                # Select a random position, ensuring no overlap
                max_attempts = 10
                attempt = 0
                position_found = False
                
                while attempt < max_attempts and not position_found:
                    # Generate random position
                    x = random.randint(0, bg_width - new_width)
                    y = random.randint(0, bg_height - new_height)
                    
                    # Check if position overlaps with existing signs
                    overlap = False
                    for px, py, pw, ph in positions:
                        # Check for intersection
                        if (x < px + pw and x + new_width > px and
                            y < py + ph and y + new_height > py):
                            overlap = True
                            break
                            
                    if not overlap:
                        position_found = True
                        positions.append((x, y, new_width, new_height))
                        
                        # Create a mask for smooth blending
                        mask = np.ones(resized.shape, dtype=np.uint8) * 255
                        
                        # Place the sign on the background
                        roi = bg[y:y+new_height, x:x+new_width]
                        
                        # Blend the sign with the background
                        blended = cv2.seamlessClone(
                            resized, roi, mask, (new_width//2, new_height//2),
                            cv2.NORMAL_CLONE
                        )
                        
                        bg[y:y+new_height, x:x+new_width] = blended
                        
                        # Add annotation for the sign
                        with open(os.path.join(output_dir, f"composite_{i+1}.txt"), 'a') as f:
                            # Format: sign_type,category,x,y,width,height
                            f.write(f"{sign_type},{category},{x},{y},{new_width},{new_height}\n")
                    
                    attempt += 1
            
            # Save the composite image
            output_path = os.path.join(output_dir, f"composite_{i+1}.jpg")
            cv2.imwrite(output_path, bg)
            composite_paths.append(output_path)
            
        return composite_paths

def main():
    augmentor = DataAugmentor()
    
    print("Moroccan Road Sign Data Augmentation Tool")
    print("----------------------------------------")
    print("1. Augment existing templates")
    print("2. Generate synthetic backgrounds")
    print("3. Create composite images with signs")
    print("4. Do all of the above")
    print("0. Exit")
    
    choice = input("Enter your choice: ")
    
    if choice == '1':
        # Augment templates
        print("\nAugmenting templates...")
        count = augmentor.augment_all_templates(num_per_template=5)
        print(f"Generated {count} augmented images")
        
    elif choice == '2':
        # Generate backgrounds
        print("\nGenerating synthetic backgrounds...")
        count = len(augmentor.generate_synthetic_backgrounds(num_backgrounds=10))
        print(f"Generated {count} background images")
        
    elif choice == '3':
        # Create composites
        print("\nCreating composite images...")
        count = len(augmentor.composite_signs_on_backgrounds(num_composites=15))
        print(f"Generated {count} composite images with annotations")
        
    elif choice == '4':
        # Do all
        print("\nAugmenting templates...")
        augmentor.augment_all_templates(num_per_template=5)
        
        print("\nGenerating synthetic backgrounds...")
        augmentor.generate_synthetic_backgrounds(num_backgrounds=10)
        
        print("\nCreating composite images...")
        count = len(augmentor.composite_signs_on_backgrounds(num_composites=15))
        print(f"Generated {count} composite images with annotations")
        
    print("\nDone!")

if __name__ == "__main__":
    main()