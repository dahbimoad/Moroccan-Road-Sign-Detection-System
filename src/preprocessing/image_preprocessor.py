import cv2
import numpy as np

class ImagePreprocessor:
    """
    Image preprocessor for road sign detection.
    
    This class handles preprocessing of input images to enhance the detection
    of Moroccan road signs under various lighting and environmental conditions.
    """
    
    def __init__(self):
        """Initialize the preprocessor with default parameters"""
        # Default parameters
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
    def process(self, image):
        """
        Process an image to improve sign detection
        
        Args:
            image: Input BGR image
            
        Returns:
            Processed image
        """
        if image is None or image.size == 0:
            return None
            
        # Apply a series of preprocessing steps
        image = self.adjust_gamma(image)
        image = self.enhance_contrast(image)
        image = self.reduce_noise(image)
        
        return image
        
    def adjust_gamma(self, image, gamma=1.2):
        """
        Adjust gamma to handle different lighting conditions
        
        Args:
            image: Input BGR image
            gamma: Gamma value (>1: darker, <1: brighter)
            
        Returns:
            Gamma-corrected image
        """
        # Build a lookup table mapping pixel values [0, 255] to adjusted values
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)
        ]).astype(np.uint8)
        
        # Apply gamma correction using the lookup table
        return cv2.LUT(image, table)
        
    def enhance_contrast(self, image):
        """
        Enhance image contrast using CLAHE on the L channel in LAB color space
        
        Args:
            image: Input BGR image
            
        Returns:
            Contrast-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split LAB channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l = self.clahe.apply(l)
        
        # Merge channels back
        enhanced_lab = cv2.merge((l, a, b))
        
        # Convert back to BGR
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_bgr
        
    def reduce_noise(self, image):
        """
        Apply noise reduction while preserving edges
        
        Args:
            image: Input BGR image
            
        Returns:
            Denoised image
        """
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        return denoised
        
    def detect_edges(self, image):
        """
        Detect edges in an image
        
        Args:
            image: Input BGR image
            
        Returns:
            Edge image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detector
        edges = cv2.Canny(blurred, 50, 150)
        
        return edges
        
    def generate_color_masks(self, image):
        """
        Generate color masks for visualization and debugging
        
        Args:
            image: Input BGR image
            
        Returns:
            Visualization of color segmentation
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for road signs (optimized for Morocco)
        lower_red1 = np.array([0, 100, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 70])
        upper_red2 = np.array([180, 255, 255])
        
        lower_blue = np.array([100, 80, 70])
        upper_blue = np.array([140, 255, 255])
        
        lower_yellow = np.array([15, 80, 70])
        upper_yellow = np.array([35, 255, 255])
        
        # Create masks
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Create colored masks
        red_viz = cv2.bitwise_and(image, image, mask=mask_red)
        blue_viz = cv2.bitwise_and(image, image, mask=mask_blue)
        yellow_viz = cv2.bitwise_and(image, image, mask=mask_yellow)
        
        # Create a combined visualization
        combined = np.zeros_like(image)
        combined[mask_red > 0] = (0, 0, 255)    # Red
        combined[mask_blue > 0] = (255, 0, 0)   # Blue
        combined[mask_yellow > 0] = (0, 255, 255)  # Yellow
        
        # Put original and masked images side by side
        h, w = image.shape[:2]
        
        # Create output visualization
        output = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        output[0:h, 0:w] = image
        output[0:h, w:w*2] = combined
        output[h:h*2, 0:w] = red_viz
        output[h:h*2, w:w*2] = np.hstack((blue_viz, yellow_viz))
        
        return output
        
    def extract_roi(self, image, roi):
        """
        Extract a region of interest from an image
        
        Args:
            image: Input image
            roi: Tuple (x, y, w, h) defining the region of interest
            
        Returns:
            Extracted ROI
        """
        x, y, w, h = roi
        return image[y:y+h, x:x+w]
        
    def adaptive_threshold(self, image):
        """
        Apply adaptive thresholding to handle variable lighting
        
        Args:
            image: Input grayscale image
            
        Returns:
            Binary threshold image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply adaptive thresholding
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
    def enhance_for_desert_conditions(self, image):
        """
        Enhance image for desert conditions (high brightness, low contrast)
        common in some Moroccan regions
        
        Args:
            image: Input BGR image
            
        Returns:
            Enhanced image
        """
        # Convert to LAB for better control over luminance
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Check if image is overexposed
        avg_lightness = np.mean(l)
        
        if avg_lightness > 150:  # Image is bright/overexposed
            # Reduce brightness
            l = cv2.subtract(l, 30)
            
            # Increase contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def enhance_for_urban_conditions(self, image):
        """
        Enhance image for urban conditions (shadows, varied lighting)
        common in Moroccan cities
        
        Args:
            image: Input BGR image
            
        Returns:
            Enhanced image
        """
        # Apply shadow removal technique
        rgb_planes = cv2.split(image)
        
        result_planes = []
        for plane in rgb_planes:
            dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            result_planes.append(diff_img)
            
        shadow_removed = cv2.merge(result_planes)
        
        # Enhance contrast
        enhanced = self.enhance_contrast(shadow_removed)
        
        return enhanced
        
    def detect_by_environment(self, image):
        """
        Automatically detect environment and apply appropriate enhancement
        
        Args:
            image: Input BGR image
            
        Returns:
            Enhanced image
        """
        # Convert to HSV for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Calculate metrics
        avg_v = np.mean(v)
        std_v = np.std(v)
        avg_s = np.mean(s)
        
        # Determine environment
        if avg_v > 170 and std_v < 50:
            # Bright, low contrast - likely desert/rural
            return self.enhance_for_desert_conditions(image)
        elif (avg_v < 120 or std_v > 60) and avg_s > 30:
            # Darker with higher saturation variance - likely urban
            return self.enhance_for_urban_conditions(image)
        else:
            # Default processing
            return self.process(image)
