import cv2
import numpy as np
from src.preprocessing.image_preprocessor import ImagePreprocessor

class SignDetector:
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        
    def detect(self, image):
        """
        Detect road signs in the image
        
        Args:
            image: Input BGR image
            
        Returns:
            List of (x, y, width, height) tuples for detected signs
        """
        # Apply automatic environment-specific preprocessing
        enhanced_image = self.preprocessor.detect_by_environment(image)
        
        # Get color segmentation masks
        color_masks = self.segment_by_color(enhanced_image)
        
        # Combine results from different detection methods
        detections = []
        
        # Method 1: Color-based detection + shape analysis
        color_detections = self.detect_by_color_and_shape(enhanced_image, color_masks)
        detections.extend(color_detections)
        
        # Method 2: Edge-based detection
        edge_detections = self.detect_by_edges(enhanced_image)
        detections.extend(edge_detections)
        
        # Method 3: Circle detection (for circular signs)
        circle_detections = self.detect_circles(enhanced_image)
        detections.extend(circle_detections)
        
        # Method 4: Triangle detection (for warning signs)
        triangle_detections = self.detect_triangles(enhanced_image)
        detections.extend(triangle_detections)
        
        # Method 5: Rectangle detection (for informational signs)
        rectangle_detections = self.detect_rectangles(enhanced_image)
        detections.extend(rectangle_detections)
        
        # Method 6: Octagon detection (for stop signs)
        octagon_detections = self.detect_octagons(enhanced_image)
        detections.extend(octagon_detections)
        
        # Apply non-maximum suppression to remove overlapping boxes
        if len(detections) > 1:
            detections = self.non_max_suppression(detections, overlap_threshold=0.3)
            
        # Filter out unlikely candidates based on sign properties
        detections = self.filter_candidates(enhanced_image, detections)
        
        return detections
    
    def segment_by_color(self, image):
        """
        Segment the image by colors relevant to Moroccan road signs
        
        Args:
            image: Input BGR image
            
        Returns:
            Dictionary of binary masks for different colors
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for Moroccan road signs
        # Red (two ranges because hue wraps around)
        lower_red1 = np.array([0, 100, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 70])
        upper_red2 = np.array([180, 255, 255])
        
        # Blue
        lower_blue = np.array([100, 80, 70])
        upper_blue = np.array([140, 255, 255])
        
        # Yellow
        lower_yellow = np.array([15, 80, 70])
        upper_yellow = np.array([35, 255, 255])
        
        # White (for speed limit signs)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        # Create color masks
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # Apply morphological operations to clean up masks
        kernel = np.ones((5, 5), np.uint8)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
        
        return {
            'red': mask_red,
            'blue': mask_blue,
            'yellow': mask_yellow,
            'white': mask_white
        }
        
    def detect_by_color_and_shape(self, image, color_masks):
        """
        Detect signs based on color segmentation and shape analysis
        
        Args:
            image: Input BGR image
            color_masks: Dictionary of binary masks for different colors
            
        Returns:
            List of (x, y, width, height) tuples for detected signs
        """
        detections = []
        
        # For each color mask
        for color, mask in color_masks.items():
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process each contour
            for contour in contours:
                # Filter small contours
                area = cv2.contourArea(contour)
                if area < 500:  # Minimum area threshold
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (most traffic signs are roughly square)
                aspect_ratio = float(w) / h
                if aspect_ratio < 0.5 or aspect_ratio > 1.5:
                    continue
                
                # Check solidity (ratio of contour area to convex hull area)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = float(area) / hull_area
                    if solidity < 0.7:  # Most signs have high solidity
                        continue
                
                # Add to detections
                detections.append((x, y, w, h))
                
        return detections
        
    def detect_by_edges(self, image):
        """
        Detect signs based on edge detection and shape analysis
        
        Args:
            image: Input BGR image
            
        Returns:
            List of (x, y, width, height) tuples for detected signs
        """
        detections = []
        
        # Get edge image
        edges = self.preprocessor.detect_edges(image)
        
        # Apply morphological operations to strengthen edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours from edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour
        for contour in contours:
            # Filter small contours
            area = cv2.contourArea(contour)
            if area < 500:  # Minimum area threshold
                continue
            
            # Approximate contour to detect shape
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if the shape could be a road sign (triangle, circle, rectangle)
            num_vertices = len(approx)
            if num_vertices >= 3 and num_vertices <= 12:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (most traffic signs are roughly square)
                aspect_ratio = float(w) / h
                if aspect_ratio < 0.5 or aspect_ratio > 1.5:
                    continue
                
                # Add to detections
                detections.append((x, y, w, h))
        
        return detections
    
    def detect_circles(self, image):
        """
        Detect circular signs using Hough Circles and contour analysis
        
        Args:
            image: Input BGR image
            
        Returns:
            List of (x, y, width, height) tuples for detected signs
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply blur to improve circle detection
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles using Hough transform
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=100,
            param1=100,  # Upper threshold for Canny edge detector
            param2=30,   # Threshold for center detection
            minRadius=20, 
            maxRadius=150
        )
        
        detections = []
        if circles is not None:
            # Convert circle parameters to integers
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Convert circle to bounding rectangle
                x_rect = x - r
                y_rect = y - r
                w = h = 2 * r
                
                # Ensure coordinates are within image bounds
                if x_rect >= 0 and y_rect >= 0 and x_rect + w < image.shape[1] and y_rect + h < image.shape[0]:
                    # ROI for verification
                    roi = image[y_rect:y_rect+h, x_rect:x_rect+w]
                    
                    # Check if the circle contains red/blue/yellow colors (common for signs)
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    
                    # Check for sign colors
                    red_pixels = cv2.inRange(hsv_roi, np.array([0, 100, 100]), np.array([10, 255, 255])) \
                              | cv2.inRange(hsv_roi, np.array([160, 100, 100]), np.array([180, 255, 255]))
                    blue_pixels = cv2.inRange(hsv_roi, np.array([100, 100, 100]), np.array([140, 255, 255]))
                    
                    red_ratio = np.count_nonzero(red_pixels) / (w * h)
                    blue_ratio = np.count_nonzero(blue_pixels) / (w * h)
                    
                    # If enough colored pixels, it's likely a sign
                    if red_ratio > 0.15 or blue_ratio > 0.15:
                        detections.append((x_rect, y_rect, w, h))
        
        return detections
    
    def detect_triangles(self, image):
        """
        Detect triangular warning signs with improved algorithm
        
        Args:
            image: Input BGR image
            
        Returns:
            List of (x, y, width, height) tuples for detected triangular signs
        """
        detections = []
        
        # Convert to grayscale and apply adaptive thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get HSV for color verification
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < 1000:
                continue
                
            # Approximate the contour
            epsilon = 0.03 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's a triangle (3 vertices)
            if len(approx) == 3:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(approx)
                
                # Check if the triangle has appropriate proportions
                aspect_ratio = float(w) / h
                if 0.7 <= aspect_ratio <= 1.3:
                    # Create a mask for the triangle
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                    cv2.drawContours(mask, [approx], 0, 255, -1)
                    
                    # Check if the triangle contains typical warning sign colors (yellow/red)
                    roi_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
                    yellow_mask = cv2.inRange(roi_hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))
                    red_mask1 = cv2.inRange(roi_hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
                    red_mask2 = cv2.inRange(roi_hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
                    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
                    
                    # Check color ratios
                    yellow_ratio = np.count_nonzero(yellow_mask) / area if area > 0 else 0
                    red_ratio = np.count_nonzero(red_mask) / area if area > 0 else 0
                    
                    # Triangles with significant yellow or red are likely signs
                    if yellow_ratio > 0.2 or red_ratio > 0.2:
                        detections.append((x, y, w, h))
        
        return detections
    
    def detect_rectangles(self, image):
        """
        Detect rectangular information signs
        
        Args:
            image: Input BGR image
            
        Returns:
            List of (x, y, width, height) tuples for detected rectangular signs
        """
        detections = []
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get HSV for color verification
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < 1000:
                continue
                
            # Approximate the contour
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's a rectangle (4 vertices)
            if len(approx) == 4:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(approx)
                
                # Check if it's a rectangle with appropriate proportions
                aspect_ratio = float(w) / h
                if 0.5 <= aspect_ratio <= 2.0:
                    # Create a mask for the rectangle
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                    cv2.drawContours(mask, [approx], 0, 255, -1)
                    
                    # Check if the rectangle contains typical sign colors (blue/white)
                    roi_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
                    blue_mask = cv2.inRange(roi_hsv, np.array([100, 80, 80]), np.array([140, 255, 255]))
                    white_mask = cv2.inRange(roi_hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
                    
                    # Check color ratios
                    blue_ratio = np.count_nonzero(blue_mask) / area if area > 0 else 0
                    white_ratio = np.count_nonzero(white_mask) / area if area > 0 else 0
                    
                    # Rectangles with significant blue are likely information signs
                    if blue_ratio > 0.3 or white_ratio > 0.5:
                        detections.append((x, y, w, h))
        
        return detections
    
    def detect_octagons(self, image):
        """
        Detect octagonal stop signs
        
        Args:
            image: Input BGR image
            
        Returns:
            List of (x, y, width, height) tuples for detected stop signs
        """
        detections = []
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get HSV for color verification
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < 1000:
                continue
                
            # Approximate the contour
            epsilon = 0.03 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it has approximately 8 vertices (octagon)
            if 7 <= len(approx) <= 9:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(approx)
                
                # Check if it's roughly square (stop signs are generally symmetric)
                aspect_ratio = float(w) / h
                if 0.8 <= aspect_ratio <= 1.2:
                    # Create a mask for the octagon
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                    cv2.drawContours(mask, [approx], 0, 255, -1)
                    
                    # Check if the octagon is predominantly red (stop sign color)
                    roi_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
                    red_mask1 = cv2.inRange(roi_hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
                    red_mask2 = cv2.inRange(roi_hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
                    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
                    
                    # Check color ratio
                    red_ratio = np.count_nonzero(red_mask) / area if area > 0 else 0
                    
                    # Octagons with significant red are likely stop signs
                    if red_ratio > 0.3:
                        detections.append((x, y, w, h))
        
        return detections
        
    def filter_candidates(self, image, candidates):
        """
        Filter detection candidates based on additional criteria
        
        Args:
            image: Input BGR image
            candidates: List of (x, y, w, h) detection candidates
            
        Returns:
            Filtered list of candidates
        """
        filtered = []
        
        # Image dimensions
        height, width = image.shape[:2]
        
        for (x, y, w, h) in candidates:
            # Filter out very small detections
            if w < 20 or h < 20:
                continue
                
            # Filter out very large detections
            if w > width * 0.5 or h > height * 0.5:
                continue
                
            # Extract ROI
            roi = image[y:y+h, x:x+w]
            
            # Check if ROI has enough color variance (signs have distinct colors)
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            color_std = np.std(hsv_roi[:, :, 1])  # Standard deviation of saturation
            
            if color_std < 10:  # Very low variance might be a false positive
                continue
                
            # Add to filtered detections
            filtered.append((x, y, w, h))
            
        return filtered
        
    def non_max_suppression(self, boxes, overlap_threshold=0.3):
        """
        Apply non-maximum suppression to avoid duplicate detections
        
        Args:
            boxes: List of (x, y, width, height) tuples
            overlap_threshold: Maximum allowed overlap ratio
            
        Returns:
            Filtered list of boxes
        """
        if len(boxes) == 0:
            return []
        
        # Convert to numpy array for easier manipulation
        boxes = np.array(boxes)
        
        # Initialize the list of picked indexes
        pick = []
        
        # Grab the coordinates of the boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        # Compute the area of the boxes
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Sort the boxes by the bottom-right y-coordinate
        idxs = np.argsort(y2)
        
        # Keep looping while some indexes still remain in the index list
        while len(idxs) > 0:
            # Grab the last index in the indexes list and add it to the list of picked
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            # Find the largest (x, y) coordinates for the start of the bbox and
            # the smallest (x, y) coordinates for the end of the bbox
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            # Compute the width and height of the bbox
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            # Compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
            
            # Delete all indexes from the index list that have an overlap greater than the threshold
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))
            
        # Return only the bounding boxes that were picked
        return boxes[pick].tolist()
