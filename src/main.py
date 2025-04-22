import argparse
import cv2
import numpy as np
import time
import os
import json
from pathlib import Path

# Add project root to path for imports
import sys
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.detection.sign_detector import SignDetector
from src.recognition.sign_classifier import SignClassifier
from src.utils.template_manager import TemplateManager

def parse_arguments():
    parser = argparse.ArgumentParser(description="Moroccan Road Sign Detection System")
    parser.add_argument("--mode", choices=["real-time", "video", "image", "batch"], 
                      default="real-time", help="Processing mode")
    parser.add_argument("--input", help="Path to input video, image file or directory (for batch mode)")
    parser.add_argument("--show-steps", action="store_true", 
                      help="Display intermediate processing steps")
    parser.add_argument("--save-output", action="store_true",
                      help="Save processed frames to output directory")
    parser.add_argument("--output-dir", default="output",
                      help="Directory to save output frames")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug visualization")
    parser.add_argument("--detection-threshold", type=float, default=0.5,
                      help="Confidence threshold for sign detection")
    parser.add_argument("--generate-report", action="store_true",
                      help="Generate detection report")
    parser.add_argument("--environment", choices=["auto", "urban", "desert", "normal"],
                      default="auto", help="Specify environment type for image processing")
    parser.add_argument("--extract-templates", action="store_true",
                      help="Extract detected signs as templates")
    parser.add_argument("--gpu", action="store_true",
                      help="Use GPU acceleration if available")
    return parser.parse_args()

class RoadSignDetectionSystem:
    def __init__(self, show_steps=False, debug=False, detection_threshold=0.5, 
                environment="auto", use_gpu=False):
        self.preprocessor = ImagePreprocessor()
        self.detector = SignDetector()
        self.classifier = SignClassifier()
        self.template_manager = TemplateManager()
        self.show_steps = show_steps
        self.debug = debug
        self.detection_threshold = detection_threshold
        self.environment = environment
        self.previous_detections = []
        self.detection_history = []  # Track detections over time for stability
        self.detected_signs = {}  # Dictionary to track unique detected signs
        self.detection_counts = {}  # Counter for each sign type
        self.processing_times = []  # List to track processing times
        
        # Set up GPU usage if requested and available
        self.use_gpu = use_gpu
        if use_gpu:
            try:
                cv2.setUseOptimized(True)
                if cv2.useOptimized():
                    print("OpenCV GPU optimization enabled")
                    
                # Check if CUDA is available
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    print(f"CUDA-enabled devices found: {cv2.cuda.getCudaEnabledDeviceCount()}")
                    self.use_cuda = True
                else:
                    print("No CUDA-enabled devices found")
                    self.use_cuda = False
            except Exception as e:
                print(f"GPU acceleration error: {e}")
                self.use_gpu = False
                self.use_cuda = False
        else:
            self.use_cuda = False
            
    def process_frame(self, frame):
        if frame is None or frame.size == 0:
            print("Error: Empty frame received")
            return None
            
        # Step 1: Preprocess the image
        start_time = time.time()
        
        # Apply environment-specific processing
        if self.environment == "auto":
            processed_image = self.preprocessor.detect_by_environment(frame)
        elif self.environment == "urban":
            processed_image = self.preprocessor.enhance_for_urban_conditions(frame)
        elif self.environment == "desert":
            processed_image = self.preprocessor.enhance_for_desert_conditions(frame)
        else:
            processed_image = self.preprocessor.process(frame)
        
        if self.show_steps:
            cv2.imshow("Preprocessed", processed_image)
            
        # Step 2: Detect road signs in the image
        detections = self.detector.detect(processed_image)
        
        # Apply temporal filtering to reduce false positives
        if len(self.previous_detections) > 0:
            detections = self.apply_temporal_filtering(detections)
            
        # Step 3: Classify detected signs
        results = []
        debug_images = {}
        
        for x, y, w, h in detections:
            # Extract the region of interest
            if y >= 0 and y+h <= frame.shape[0] and x >= 0 and x+w <= frame.shape[1]:
                roi = frame[y:y+h, x:x+w]
                
                if roi.size == 0:
                    continue
                    
                # Classify the sign
                sign_type, confidence = self.classifier.classify(roi)
                
                # Only include detections above threshold
                if confidence >= self.detection_threshold:
                    results.append((x, y, w, h, sign_type, confidence))
                    
                    # Get the sign meaning
                    sign_meaning = self.classifier.get_sign_meaning(sign_type)
                    sign_category = self.classifier.get_sign_category(sign_type)
                    
                    # Update detection counts
                    if sign_type not in self.detection_counts:
                        self.detection_counts[sign_type] = 1
                    else:
                        self.detection_counts[sign_type] += 1
                    
                    # Store unique detection
                    sign_key = f"{sign_type}_{self.detection_counts[sign_type]}"
                    if sign_key not in self.detected_signs:
                        self.detected_signs[sign_key] = {
                            'type': sign_type,
                            'category': sign_category,
                            'meaning': sign_meaning,
                            'confidence': confidence,
                            'roi': roi,
                            'position': (x, y, w, h),
                            'first_detected': time.time()
                        }
                    
                    # Store for debugging
                    if self.debug:
                        debug_roi = cv2.resize(roi, (64, 64))
                        debug_images[(x, y, w, h)] = {
                            'roi': debug_roi,
                            'type': sign_type,
                            'category': sign_category,
                            'meaning': sign_meaning,
                            'confidence': confidence
                        }
        
        # Store current detections for next frame
        self.previous_detections = detections
            
        # Step 4: Draw results on the original frame
        result_frame = frame.copy()
        result_frame = self.draw_results(result_frame, results)
        
        # Add debug visualization if enabled
        if self.debug and debug_images:
            result_frame = self.add_debug_visualization(result_frame, debug_images)
            
        # Show color masks if in debug mode
        if self.debug:
            color_viz = self.preprocessor.generate_color_masks(processed_image)
            cv2.imshow("Color Segmentation", color_viz)
            
            # Show edge detection
            edges = self.preprocessor.detect_edges(processed_image)
            cv2.imshow("Edge Detection", edges)
            
        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return result_frame
    
    def apply_temporal_filtering(self, current_detections):
        """
        Apply temporal filtering to reduce false positives
        """
        # Create a list to keep track of matched detections
        matched_detections = []
        
        # For each current detection
        for x, y, w, h in current_detections:
            matched = False
            
            # Check if it overlaps with any previous detection
            for prev_x, prev_y, prev_w, prev_h in self.previous_detections:
                # Calculate IoU
                intersection_x = max(x, prev_x)
                intersection_y = max(y, prev_y)
                intersection_w = min(x + w, prev_x + prev_w) - intersection_x
                intersection_h = min(y + h, prev_y + prev_h) - intersection_y
                
                if intersection_w > 0 and intersection_h > 0:
                    intersection_area = intersection_w * intersection_h
                    current_area = w * h
                    prev_area = prev_w * prev_h
                    union_area = current_area + prev_area - intersection_area
                    iou = intersection_area / union_area
                    
                    if iou > 0.3:  # Threshold for considering same detection
                        matched = True
                        break
            
            if matched:
                matched_detections.append((x, y, w, h))
        
        return matched_detections
    
    def draw_results(self, frame, results):
        """Draw detection results on the frame with Moroccan-specific styling"""
        # Add a semi-transparent overlay for text at the bottom of the frame
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, frame.shape[0] - 40), 
                     (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Add summary information at the bottom
        sign_count = len(results)
        cv2.putText(frame, f"Signs Detected: {sign_count}", 
                   (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        
        # Draw a border around the frame (Moroccan flag colors)
        border_color = (0, 135, 81)  # Green (Moroccan flag color)
        thickness = 3
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), 
                     border_color, thickness)
        
        # Process each detection
        for x, y, w, h, sign_type, confidence in results:
            # Determine color based on sign type and category
            category = self.classifier.get_sign_category(sign_type)
            
            if category == "regulatory":
                box_color = (0, 0, 255)  # Red for regulatory signs
            elif category == "warning":
                box_color = (0, 165, 255)  # Orange for warning signs
            elif category == "information":
                box_color = (255, 0, 0)  # Blue for information signs
            else:
                box_color = (0, 255, 0)  # Green for other signs
            
            # Draw bounding box with more attractive style
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            
            # Add a small colored header above the box
            header_height = 20
            cv2.rectangle(frame, (x, y-header_height), (x+w, y), box_color, -1)
            
            # Get sign meaning
            sign_meaning = self.classifier.get_sign_meaning(sign_type)
            
            # Add sign type in the header
            cv2.putText(frame, sign_type.replace("_", " ").title(), 
                       (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw label with confidence and sign meaning
            label = f"{sign_meaning} ({confidence:.2f})"
            
            # Create background for better text visibility
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x, y+h), 
                         (x + text_size[0], y+h+text_size[1]+10), (0, 0, 0), -1)
            
            cv2.putText(frame, label, (x, y+h+text_size[1]+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return frame
    
    def add_debug_visualization(self, frame, debug_images):
        """Add debug visualization with Moroccan-styled info panel"""
        # Create a sidebar for debug info
        height = frame.shape[0]
        sidebar_width = 220
        sidebar = np.ones((height, sidebar_width, 3), dtype=np.uint8) * 30  # Dark background
        
        # Add Moroccan flag-inspired design elements
        cv2.rectangle(sidebar, (0, 0), (sidebar_width-1, height-1), (0, 135, 81), 2)  # Green border
        
        # Add header
        cv2.rectangle(sidebar, (0, 0), (sidebar_width, 40), (0, 135, 81), -1)  # Green header
        cv2.putText(sidebar, "Detection Details", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add detected signs to the sidebar
        y_offset = 50
        for idx, (key, data) in enumerate(debug_images.items()):
            if idx >= 5:  # Limit to 5 signs to avoid overcrowding
                break
                
            # Add thumbnail
            roi = data['roi']
            h, w = roi.shape[:2]
            
            if y_offset + h > height - 50:
                continue
                
            # Add thumbnail border
            cv2.rectangle(sidebar, (10-2, y_offset-2), (10+w+2, y_offset+h+2), (255, 255, 255), 1)
            sidebar[y_offset:y_offset+h, 10:10+w] = roi
            
            # Add category color indicator
            category = data['category']
            if category == "regulatory":
                cat_color = (0, 0, 255)  # Red
                cat_text = "Regulatory"
            elif category == "warning":
                cat_color = (0, 165, 255)  # Orange
                cat_text = "Warning"
            elif category == "information":
                cat_color = (255, 0, 0)  # Blue
                cat_text = "Information"
            else:
                cat_color = (0, 255, 0)  # Green
                cat_text = "Other"
                
            cv2.rectangle(sidebar, (10+w+10, y_offset), (10+w+20, y_offset+h), cat_color, -1)
            
            # Add text details
            cv2.putText(sidebar, f"Type: {data['type']}", (10, y_offset+h+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            cv2.putText(sidebar, f"Conf: {data['confidence']:.2f}", (10, y_offset+h+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            cv2.putText(sidebar, cat_text, (10, y_offset+h+45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, cat_color, 1)
            
            y_offset += h + 60
        
        # Add performance metrics at the bottom
        avg_time = np.mean(self.processing_times[-100:]) if self.processing_times else 0
        fps = 1 / avg_time if avg_time > 0 else 0
        
        cv2.rectangle(sidebar, (0, height-70), (sidebar_width, height), (40, 40, 40), -1)
        cv2.putText(sidebar, f"FPS: {fps:.1f}", (10, height-45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.putText(sidebar, f"Time: {avg_time*1000:.1f}ms", (10, height-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Combine frame and sidebar
        result = np.zeros((height, frame.shape[1] + sidebar_width, 3), dtype=np.uint8)
        result[:, :frame.shape[1]] = frame
        result[:, frame.shape[1]:] = sidebar
        
        return result
        
    def extract_sign_templates(self, output_dir=None):
        """Extract detected signs as templates for future training"""
        if output_dir is None:
            output_dir = os.path.join(root_dir, "data", "templates")
            
        extracted_count = 0
        
        for sign_key, sign_data in self.detected_signs.items():
            sign_type = sign_data['type']
            category = sign_data['category']
            roi = sign_data['roi']
            confidence = sign_data['confidence']
            
            if confidence < 0.7:  # Only extract high-confidence detections
                continue
                
            # Create category and sign type directories
            category_dir = os.path.join(output_dir, category)
            sign_type_dir = os.path.join(category_dir, sign_type)
            
            os.makedirs(sign_type_dir, exist_ok=True)
            
            # Save the template
            timestamp = int(time.time())
            filename = f"{sign_type}_{timestamp}.jpg"
            filepath = os.path.join(sign_type_dir, filename)
            
            cv2.imwrite(filepath, roi)
            extracted_count += 1
            
            print(f"Extracted template: {filepath}")
            
        return extracted_count
        
    def generate_detection_report(self, output_dir=None):
        """Generate a JSON report of detected signs"""
        if output_dir is None:
            output_dir = "output"
            
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            "detection_summary": {
                "total_signs_detected": len(self.detected_signs),
                "sign_types_detected": {},
                "average_processing_time_ms": np.mean(self.processing_times) * 1000 if self.processing_times else 0,
                "detection_threshold": self.detection_threshold,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "detected_signs": {}
        }
        
        # Count sign types
        sign_type_counts = {}
        for sign_key, sign_data in self.detected_signs.items():
            sign_type = sign_data['type']
            if sign_type not in sign_type_counts:
                sign_type_counts[sign_type] = 1
            else:
                sign_type_counts[sign_type] += 1
                
        report["detection_summary"]["sign_types_detected"] = sign_type_counts
        
        # Add detailed information about each detection
        for sign_key, sign_data in self.detected_signs.items():
            report["detected_signs"][sign_key] = {
                "type": sign_data['type'],
                "category": sign_data['category'],
                "meaning": sign_data['meaning'],
                "confidence": sign_data['confidence'],
                "position": sign_data['position'],
                "first_detected": sign_data['first_detected']
            }
            
        # Save the report
        timestamp = int(time.time())
        report_path = os.path.join(output_dir, f"detection_report_{timestamp}.json")
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"Detection report saved to {report_path}")
        
        # Also save a summary image
        if self.detected_signs:
            self.save_detection_summary_image(output_dir)
            
        return report_path
        
    def save_detection_summary_image(self, output_dir):
        """Save a summary image showing all detected signs"""
        # Determine grid size based on number of signs
        num_signs = len(self.detected_signs)
        if num_signs == 0:
            return None
            
        grid_size = min(5, max(1, int(np.ceil(np.sqrt(num_signs)))))
        
        # Create a blank canvas
        cell_size = 120
        padding = 10
        grid_width = grid_size * (cell_size + padding) + padding
        grid_height = ((num_signs + grid_size - 1) // grid_size) * (cell_size + padding) + padding
        grid_height = max(grid_height, 300)  # Minimum height
        
        canvas = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Add header
        cv2.rectangle(canvas, (0, 0), (grid_width, 50), (0, 135, 81), -1)  # Green header (Moroccan flag color)
        cv2.putText(canvas, "Moroccan Road Sign Detection Summary", (padding, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(canvas, timestamp, (grid_width - 200, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add detected signs to the grid
        x, y = padding, 50 + padding
        for sign_key, sign_data in self.detected_signs.items():
            # Get sign ROI and resize
            roi = sign_data['roi']
            roi_resized = cv2.resize(roi, (cell_size, cell_size))
            
            # Get category for color coding
            category = sign_data['category']
            if category == "regulatory":
                color = (0, 0, 255)  # Red
            elif category == "warning":
                color = (0, 165, 255)  # Orange
            elif category == "information":
                color = (255, 0, 0)  # Blue
            else:
                color = (0, 255, 0)  # Green
                
            # Draw border with category color
            cv2.rectangle(canvas, (x-2, y-2), (x+cell_size+2, y+cell_size+2), color, 2)
            
            # Place sign image
            canvas[y:y+cell_size, x:x+cell_size] = roi_resized
            
            # Add sign type below the image
            sign_type = sign_data['type'].replace("_", " ").title()
            confidence = sign_data['confidence']
            label = f"{sign_type} ({confidence:.2f})"
            
            # Add text background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(canvas, (x, y+cell_size+2), 
                         (x+text_size[0], y+cell_size+2+text_size[1]+4), (255, 255, 255), -1)
            
            cv2.putText(canvas, label, (x, y+cell_size+text_size[1]+4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Update position for next sign
            x += cell_size + padding
            if x + cell_size > grid_width:
                x = padding
                y += cell_size + padding + 25  # Extra space for text
                
            if y + cell_size > grid_height:
                break  # Stop if we run out of space
        
        # Add summary text
        summary_y = grid_height - 60
        cv2.rectangle(canvas, (0, summary_y), (grid_width, grid_height), (230, 230, 230), -1)
        
        cv2.putText(canvas, f"Total Signs Detected: {num_signs}", (padding, summary_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        avg_time = np.mean(self.processing_times) * 1000 if self.processing_times else 0
        cv2.putText(canvas, f"Average Processing Time: {avg_time:.1f}ms", (padding, summary_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Save the summary image
        timestamp_str = int(time.time())
        summary_path = os.path.join(output_dir, f"detection_summary_{timestamp_str}.jpg")
        cv2.imwrite(summary_path, canvas)
        
        print(f"Detection summary image saved to {summary_path}")
        
        return summary_path

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    args = parse_arguments()
    
    # Make sure output directory exists if saving output
    if args.save_output:
        ensure_dir(args.output_dir)
        
    system = RoadSignDetectionSystem(
        show_steps=args.show_steps,
        debug=args.debug,
        detection_threshold=args.detection_threshold,
        environment=args.environment,
        use_gpu=args.gpu
    )
    
    if args.mode == "real-time":
        print("Starting real-time detection with webcam...")
        print("Press 'q' to quit, 's' to save current frame, 'r' to generate report")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
            
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
                
            result_frame = system.process_frame(frame)
            
            if result_frame is None:
                continue
                
            # Calculate and display FPS
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
                cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Moroccan Road Sign Detection", result_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                # Save current frame
                ensure_dir(args.output_dir)
                filename = os.path.join(args.output_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(filename, result_frame)
                print(f"Saved frame to {filename}")
                
            elif key == ord('r'):
                # Generate detection report
                if args.generate_report:
                    ensure_dir(args.output_dir)
                    report_path = system.generate_detection_report(args.output_dir)
                    print(f"Generated report: {report_path}")
                else:
                    print("Report generation disabled. Use --generate-report flag to enable.")
                    
            frame_count += 1
                
        cap.release()
        
        # Save templates if requested
        if args.extract_templates:
            print("Extracting sign templates...")
            count = system.extract_sign_templates()
            print(f"Extracted {count} templates")
            
        # Generate report at the end if requested
        if args.generate_report:
            system.generate_detection_report(args.output_dir)
            
    elif args.mode == "video":
        if not args.input:
            print("Error: Input video file path is required for video mode")
            return
            
        if not os.path.exists(args.input):
            print(f"Error: Video file not found: {args.input}")
            return
            
        print(f"Processing video: {args.input}")
        print("Press 'q' to quit, 'p' to pause/resume")
        
        # Process video file
        cap = cv2.VideoCapture(args.input)
        
        if not cap.isOpened():
            print("Error: Could not open video file")
            return
            
        frame_count = 0
        paused = False
        
        # Get video properties for output
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer if saving output
        out = None
        if args.save_output:
            output_path = os.path.join(args.output_dir, 'output_video.mp4')
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video or error reading frame")
                    break
                    
                result_frame = system.process_frame(frame)
                
                if result_frame is not None:
                    # Resize if the debug visualization changed the dimensions
                    if result_frame.shape[1] != width or result_frame.shape[0] != height:
                        result_frame = cv2.resize(result_frame, (width, height))
                        
                    cv2.imshow("Moroccan Road Sign Detection", result_frame)
                    
                    # Save frame if requested
                    if args.save_output and out is not None:
                        out.write(result_frame)
                        
                    if args.save_output and (frame_count % 30 == 0):  # Save every 30th frame
                        filename = os.path.join(args.output_dir, f"frame_{frame_count:04d}.jpg")
                        cv2.imwrite(filename, result_frame)
                        
                    frame_count += 1
            
            # Handle keyboard input
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
                
        cap.release()
        if out is not None:
            out.release()
            
        # Save templates if requested
        if args.extract_templates:
            print("Extracting sign templates...")
            count = system.extract_sign_templates()
            print(f"Extracted {count} templates")
            
        # Generate report at the end if requested
        if args.generate_report:
            system.generate_detection_report(args.output_dir)
        
    elif args.mode == "image":
        if not args.input:
            print("Error: Input image file path is required for image mode")
            return
            
        if not os.path.exists(args.input):
            print(f"Error: Image file not found: {args.input}")
            return
            
        print(f"Processing image: {args.input}")
        
        # Process single image
        frame = cv2.imread(args.input)
        
        if frame is None:
            print("Error: Could not read image")
            return
            
        result_frame = system.process_frame(frame)
        
        if result_frame is not None:
            cv2.imshow("Moroccan Road Sign Detection", result_frame)
            
            # Save result if requested
            if args.save_output:
                base_name = os.path.basename(args.input)
                filename = os.path.join(args.output_dir, f"processed_{base_name}")
                cv2.imwrite(filename, result_frame)
                print(f"Saved processed image to {filename}")
                
            # Save templates if requested
            if args.extract_templates:
                print("Extracting sign templates...")
                count = system.extract_sign_templates()
                print(f"Extracted {count} templates")
                
            # Generate report if requested
            if args.generate_report:
                system.generate_detection_report(args.output_dir)
            
            print("Press any key to exit")
            cv2.waitKey(0)
            
    elif args.mode == "batch":
        if not args.input:
            print("Error: Input directory path is required for batch mode")
            return
            
        if not os.path.exists(args.input) or not os.path.isdir(args.input):
            print(f"Error: Input directory not found: {args.input}")
            return
            
        print(f"Processing images in directory: {args.input}")
        
        # Get all image files in directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for file in os.listdir(args.input):
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_files.append(os.path.join(args.input, file))
                
        if not image_files:
            print("No image files found in the directory")
            return
            
        print(f"Found {len(image_files)} image files")
        
        # Process each image
        processed_count = 0
        for image_file in image_files:
            print(f"Processing {image_file}...")
            
            frame = cv2.imread(image_file)
            if frame is None:
                print(f"Error: Could not read {image_file}")
                continue
                
            result_frame = system.process_frame(frame)
            
            if result_frame is not None and args.save_output:
                base_name = os.path.basename(image_file)
                output_filename = os.path.join(args.output_dir, f"processed_{base_name}")
                cv2.imwrite(output_filename, result_frame)
                processed_count += 1
                
        print(f"Processed {processed_count} images")
        
        # Save templates if requested
        if args.extract_templates:
            print("Extracting sign templates...")
            count = system.extract_sign_templates()
            print(f"Extracted {count} templates")
            
        # Generate report if requested
        if args.generate_report:
            system.generate_detection_report(args.output_dir)
        
    cv2.destroyAllWindows()
    print("Processing complete")

if __name__ == "__main__":
    main()
