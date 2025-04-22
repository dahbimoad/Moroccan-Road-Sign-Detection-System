import cv2
import numpy as np
import os
from pathlib import Path
import pytesseract

# Configure pytesseract path - update this path to where your Tesseract OCR is installed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Uncomment and set path if needed

class SignClassifier:
    def __init__(self, templates_dir=None):
        self.templates = {}
        # Extended sign meanings with Moroccan-specific signs
        self.sign_meanings = {
            # Regulatory signs
            "stop": "Stop",
            "yield": "Yield",
            "speed_limit_30": "Speed Limit 30 km/h",
            "speed_limit_40": "Speed Limit 40 km/h",
            "speed_limit_50": "Speed Limit 50 km/h",
            "speed_limit_60": "Speed Limit 60 km/h",
            "speed_limit_80": "Speed Limit 80 km/h",
            "speed_limit_100": "Speed Limit 100 km/h",
            "speed_limit_120": "Speed Limit 120 km/h",
            "no_entry": "No Entry",
            "no_parking": "No Parking",
            "no_stopping": "No Stopping",
            "no_overtaking": "No Overtaking",
            "no_pedestrians": "No Pedestrians",
            "no_horn": "No Horn",  # Common in Morocco
            "priority_road": "Priority Road",
            "give_way": "Give Way",
            "end_of_restriction": "End of Restriction",
            "end_of_speed_limit": "End of Speed Limit",
            
            # Warning signs
            "warning_general": "General Warning",
            "warning_intersection": "Intersection Warning",
            "warning_curve": "Curve Warning",
            "warning_curve_left": "Left Curve Warning",
            "warning_curve_right": "Right Curve Warning",
            "warning_roundabout": "Roundabout Ahead",
            "warning_pedestrians": "Pedestrian Crossing",
            "warning_children": "Children Crossing",
            "warning_animals": "Animal Crossing",
            "warning_bump": "Speed Bump",
            "warning_slippery": "Slippery Road",
            "warning_roadworks": "Road Works",
            "warning_traffic_lights": "Traffic Lights Ahead",
            "warning_narrow_road": "Narrow Road",
            
            # Information signs
            "information_parking": "Parking",
            "information_hospital": "Hospital",
            "information_crosswalk": "Pedestrian Crossing",
            "information_bus_stop": "Bus Stop",
            "information_taxi": "Taxi Stand",
            "information_telephone": "Telephone",
            "roundabout": "Roundabout",
            "direction_sign": "Direction",
            "city_limit": "City Limit",
            "motorway": "Motorway",
            "service_area": "Service Area",
            "exit_sign": "Exit",
            
            "unknown": "Unknown Sign"
        }
        
        # Load templates if directory is provided
        if templates_dir is None:
            root_dir = Path(__file__).resolve().parent.parent.parent
            templates_dir = os.path.join(root_dir, "data", "templates")
            
        if os.path.exists(templates_dir):
            self.load_templates(templates_dir)
            
    def load_templates(self, templates_dir):
        """
        Load template images for sign classification
        
        Args:
            templates_dir: Directory containing template images
        """
        if not os.path.exists(templates_dir):
            print(f"Warning: Templates directory not found: {templates_dir}")
            return
            
        # Load all templates from subdirectories
        for category in os.listdir(templates_dir):
            category_dir = os.path.join(templates_dir, category)
            
            if os.path.isdir(category_dir):
                # Process each sign type directory in this category
                for sign_type in os.listdir(category_dir):
                    sign_dir = os.path.join(category_dir, sign_type)
                    
                    if os.path.isdir(sign_dir):
                        # Create a key that combines category and sign type
                        key = sign_type
                        self.templates[key] = []
                        
                        # Load all template images for this sign type
                        for filename in os.listdir(sign_dir):
                            if filename.endswith(('.png', '.jpg', '.jpeg')):
                                file_path = os.path.join(sign_dir, filename)
                                template = cv2.imread(file_path)
                                
                                if template is not None:
                                    # Resize template to a standard size
                                    template = cv2.resize(template, (64, 64))
                                    self.templates[key].append(template)
                        
                        if self.templates[key]:
                            print(f"Loaded {len(self.templates[key])} templates for {key}")
                        else:
                            # Remove empty entries
                            del self.templates[key]
                
    def classify(self, sign_image):
        """
        Classify a detected road sign
        
        Args:
            sign_image: Image of the detected sign
            
        Returns:
            (sign_type, confidence) tuple
        """
        if sign_image is None or sign_image.size == 0:
            return "unknown", 0.0
            
        # Resize input image to standard size
        resized = cv2.resize(sign_image, (64, 64))
        
        # Extract features
        features = self.extract_features(resized)
        
        # Try different classification methods
        results = []
        
        # Method 1: Template matching
        if self.templates:
            sign_type, confidence = self.template_matching(resized)
            results.append((sign_type, confidence, "template"))
        
        # Method 2: Color and shape-based classification
        sign_type, confidence = self.color_shape_classification(resized)
        results.append((sign_type, confidence, "color_shape"))
        
        # Method 3: OCR for speed limit signs
        sign_type, confidence = self.ocr_classification(sign_image)
        results.append((sign_type, confidence, "ocr"))
        
        # Method 4: HOG features
        sign_type, confidence = self.hog_classification(resized)
        results.append((sign_type, confidence, "hog"))
        
        # Select the result with highest confidence
        results.sort(key=lambda x: x[1], reverse=True)
        best_result = results[0]
        
        return best_result[0], best_result[1]
        
    def template_matching(self, image):
        """
        Classify sign using template matching
        
        Args:
            image: Preprocessed sign image
            
        Returns:
            (sign_type, confidence) tuple
        """
        best_match = "unknown"
        best_score = 0.0
        
        for sign_type, templates in self.templates.items():
            for template in templates:
                # Try multiple matching methods
                methods = [
                    cv2.TM_CCOEFF_NORMED,
                    cv2.TM_CCORR_NORMED
                ]
                
                scores = []
                for method in methods:
                    # Convert both to grayscale for matching
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    
                    # Match
                    result = cv2.matchTemplate(gray_image, gray_template, method)
                    _, score, _, _ = cv2.minMaxLoc(result)
                    scores.append(score)
                
                # Use average score
                avg_score = sum(scores) / len(scores)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_match = sign_type
        
        return best_match, best_score
        
    def color_shape_classification(self, image):
        """
        Classify sign based on color and shape
        
        Args:
            image: Preprocessed sign image
            
        Returns:
            (sign_type, confidence) tuple
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate color histograms
        color_hist = self._calculate_color_histogram(hsv)
        
        # Check dominant colors
        red_ratio = np.sum(color_hist['red']) / np.sum(color_hist['all'])
        blue_ratio = np.sum(color_hist['blue']) / np.sum(color_hist['all'])
        yellow_ratio = np.sum(color_hist['yellow']) / np.sum(color_hist['all'])
        white_ratio = np.sum(color_hist['white']) / np.sum(color_hist['all'])
        
        # Convert to grayscale for shape analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours for shape analysis
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Default
        sign_type = "unknown"
        confidence = 0.3  # Base confidence
        
        # Check if any contours were found
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate shape
            epsilon = 0.04 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            num_vertices = len(approx)
            
            # Check circularity
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Classification based on color and shape
            if red_ratio > 0.3:  # Predominantly red
                # Check for circular shapes (prohibition signs)
                if circularity > 0.7:
                    # Circular prohibition sign - look for details
                    if self.detect_horizontal_line(thresh):
                        sign_type = "no_entry"
                        confidence = 0.8
                    else:
                        # Generic prohibition sign
                        sign_type = "no_entry"
                        confidence = 0.7
                
                # Check for octagonal shape (stop sign)
                elif 6 <= num_vertices <= 10:
                    sign_type = "stop"
                    confidence = 0.8
                
                # Check for triangular shape (warning/yield)
                elif num_vertices == 3 or (0.4 <= circularity <= 0.7):
                    # Triangle warning sign - inverted
                    if self.is_triangle_upright(approx):
                        sign_type = "yield"
                        confidence = 0.75
                    else:
                        sign_type = "warning_general"
                        confidence = 0.7
                        
            elif blue_ratio > 0.3:  # Predominantly blue
                if circularity > 0.7:
                    # Circular information sign
                    sign_type = "roundabout"
                    confidence = 0.65
                elif num_vertices == 4:
                    # Rectangular information sign
                    sign_type = "information_parking"
                    confidence = 0.6
                    
            elif yellow_ratio > 0.3:  # Predominantly yellow
                if num_vertices == 3 or 0.4 < circularity < 0.7:
                    # Triangle yield/warning sign
                    sign_type = "warning_general"
                    confidence = 0.7
                    
            # Check for speed limit signs (white circular with black text)
            if circularity > 0.7 and red_ratio > 0.25 and white_ratio > 0.3:
                # Speed limit sign detection will be improved with OCR
                sign_type = "speed_limit_50"  # Default, OCR will refine this
                confidence = 0.6
        
        return sign_type, confidence
    
    def ocr_classification(self, image):
        """
        Use OCR to identify text in signs (especially for speed limits)
        
        Args:
            image: Original sign image (not resized)
            
        Returns:
            (sign_type, confidence) tuple
        """
        try:
            # Prepare image for OCR
            # Resize to larger size for better OCR
            h, w = image.shape[:2]
            if h < 100 or w < 100:
                scale_factor = max(100 / h, 100 / w)
                new_h = int(h * scale_factor)
                new_w = int(w * scale_factor)
                image = cv2.resize(image, (new_w, new_h))
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding for better text extraction
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
            
            # Perform OCR
            text = pytesseract.image_to_string(thresh, config='--psm 7 -c tessedit_char_whitelist=0123456789')
            
            # Extract numbers from text
            numbers = ''.join(c for c in text if c.isdigit())
            
            # Check if we have a speed limit sign
            if numbers and len(numbers) <= 3:  # Typically speed limits are 2-3 digits
                speed = int(numbers)
                
                # Common speed limits in Morocco
                if speed <= 30:
                    return "speed_limit_30", 0.85
                elif speed <= 40:
                    return "speed_limit_40", 0.85
                elif speed <= 50:
                    return "speed_limit_50", 0.85
                elif speed <= 60:
                    return "speed_limit_60", 0.85
                elif speed <= 80:
                    return "speed_limit_80", 0.85
                elif speed <= 100:
                    return "speed_limit_100", 0.85
                elif speed <= 120:
                    return "speed_limit_120", 0.85
        
        except Exception as e:
            print(f"OCR error: {e}")
        
        return "unknown", 0.1
        
    def hog_classification(self, image):
        """
        Classify sign using HOG features
        
        Args:
            image: Preprocessed sign image
            
        Returns:
            (sign_type, confidence) tuple
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate HOG features
        win_size = (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        hog_features = hog.compute(gray)
        
        # For now, return a placeholder result
        # In a real implementation, these would be compared to known HOG features
        # or used as input to a machine learning model
        return "unknown", 0.4
        
    def extract_features(self, image):
        """
        Extract features from an image for classification
        
        Args:
            image: Input BGR image
            
        Returns:
            Dictionary of feature vectors
        """
        features = {}
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Color histogram features
        features['color_hist'] = self._calculate_color_histogram(hsv)
        
        # Edge features
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        
        # Shape features from contours
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Contour shape features
            features['area'] = cv2.contourArea(largest_contour)
            features['perimeter'] = cv2.arcLength(largest_contour, True)
            
            if features['perimeter'] > 0:
                features['circularity'] = 4 * np.pi * features['area'] / (features['perimeter'] ** 2)
            else:
                features['circularity'] = 0
                
            # Approximate shape
            epsilon = 0.04 * features['perimeter']
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            features['vertices'] = len(approx)
        else:
            # No contour found
            features['area'] = 0
            features['perimeter'] = 0
            features['circularity'] = 0
            features['vertices'] = 0
        
        return features
    
    def detect_horizontal_line(self, binary_image):
        """
        Detect a horizontal line across the middle (for no entry signs)
        
        Args:
            binary_image: Binary thresholded image
            
        Returns:
            True if horizontal line detected, False otherwise
        """
        height, width = binary_image.shape
        middle_strip = binary_image[height//3:2*height//3, :]
        
        # Check for horizontal line by summing rows
        row_sums = np.sum(middle_strip, axis=1)
        max_sum = np.max(row_sums)
        
        # If any row has a high sum (many white pixels), it's a horizontal line
        return max_sum > width * 0.6
    
    def is_triangle_upright(self, approx):
        """
        Check if a triangle is upright (yield) or upside down (warning)
        
        Args:
            approx: Approximated contour of a triangle
            
        Returns:
            True if upright, False if inverted
        """
        # Get the y-coordinates of the vertices
        y_coords = [point[0][1] for point in approx]
        
        # Find the index of the point with the minimum y-coordinate (topmost)
        top_idx = y_coords.index(min(y_coords))
        
        # Get the top point and the other two points
        top_point = approx[top_idx][0]
        
        # Calculate the center of the triangle
        center_x = np.mean([point[0][0] for point in approx])
        
        # If the top point is near the center horizontally, it's upright
        return abs(top_point[0] - center_x) < 10
        
    def _calculate_color_histogram(self, hsv_image):
        """
        Calculate color histograms for sign classification
        
        Args:
            hsv_image: HSV image
            
        Returns:
            Dictionary of color histograms
        """
        # Define color ranges - adjusted for Morocco traffic signs
        lower_red1 = np.array([0, 100, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 70])
        upper_red2 = np.array([180, 255, 255])
        
        lower_blue = np.array([100, 80, 70])
        upper_blue = np.array([140, 255, 255])
        
        lower_yellow = np.array([15, 80, 70])
        upper_yellow = np.array([35, 255, 255])
        
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        # Create masks
        mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
        mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(hsv_image, lower_white, upper_white)
        
        # Calculate histograms
        hist_red = cv2.calcHist([hsv_image], [0], mask_red, [64], [0, 180])
        hist_blue = cv2.calcHist([hsv_image], [0], mask_blue, [64], [0, 180])
        hist_yellow = cv2.calcHist([hsv_image], [0], mask_yellow, [64], [0, 180])
        hist_white = cv2.calcHist([hsv_image], [0], mask_white, [64], [0, 180])
        hist_all = cv2.calcHist([hsv_image], [0], None, [64], [0, 180])
        
        # Normalize histograms
        if np.sum(hist_red) > 0:
            hist_red /= np.sum(hist_red)
        if np.sum(hist_blue) > 0:
            hist_blue /= np.sum(hist_blue)
        if np.sum(hist_yellow) > 0:
            hist_yellow /= np.sum(hist_yellow)
        if np.sum(hist_white) > 0:
            hist_white /= np.sum(hist_white)
        if np.sum(hist_all) > 0:
            hist_all /= np.sum(hist_all)
        
        return {
            'red': hist_red,
            'blue': hist_blue,
            'yellow': hist_yellow,
            'white': hist_white,
            'all': hist_all
        }
    
    def get_sign_meaning(self, sign_type):
        """Get the meaning of a sign type"""
        return self.sign_meanings.get(sign_type, "Unknown Sign")
        
    def get_sign_category(self, sign_type):
        """Get the category of a sign type"""
        if sign_type.startswith("speed_limit") or sign_type.startswith("no_") or sign_type in ["stop", "yield", "priority_road"]:
            return "regulatory"
        elif sign_type.startswith("warning"):
            return "warning"
        elif sign_type.startswith("information") or sign_type in ["roundabout", "direction_sign"]:
            return "information"
        else:
            return "unknown"
