import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import sys
from pathlib import Path

# Add project root to path for imports
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.detection.sign_detector import SignDetector
from src.recognition.sign_classifier import SignClassifier
from src.utils.template_manager import TemplateManager

app = Flask(__name__)
app.secret_key = "moroccan_road_sign_detection_app"
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads')
app.config['RESULTS_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/results')

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize components
preprocessor = ImagePreprocessor()
template_manager = TemplateManager()
detector = SignDetector()
classifier = SignClassifier()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file was uploaded
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    # Check if the file has a name and extension
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    # Check if the file is allowed
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        try:
            results, output_img = process_image(filepath)
            
            # Save the output image with detection results
            result_filename = f"result_{filename}"
            result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
            cv2.imwrite(result_path, output_img)
            
            # Pass the results to the results page
            return render_template(
                'results.html',
                original=f"uploads/{filename}",
                result=f"results/{result_filename}",
                signs=results
            )
        
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Allowed file types are png, jpg, jpeg')
        return redirect(url_for('index'))

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image file")
    
    # Preprocess the image
    processed_img = preprocessor.preprocess(image)
    
    # Detect signs in the image
    detected_regions = detector.detect_signs(processed_img)
    
    # Classify each detected sign
    results = []
    output_img = image.copy()
    
    for i, region in enumerate(detected_regions):
        x, y, w, h = region
        sign_img = processed_img[y:y+h, x:x+w]
        
        # Classify the sign
        sign_type, confidence = classifier.classify(sign_img)
        
        # Store the result
        results.append({
            'id': i + 1,
            'type': sign_type,
            'confidence': f"{confidence:.2f}%",
            'position': {'x': x, 'y': y, 'width': w, 'height': h}
        })
        
        # Draw bounding box and label on the output image
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(output_img, (x, y), (x + w, y + h), color, 2)
        
        # Add label
        label = f"{sign_type} ({confidence:.1f}%)"
        cv2.putText(output_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return results, output_img

@app.route('/api/detect', methods=['POST'])
def api_detect():
    """API endpoint for programmatic access to sign detection"""
    
    # Check if image file is present in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    # Check if the file has a name
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    # Check if the file is allowed
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the image
            results, _ = process_image(filepath)
            
            # Return the results as JSON
            return jsonify({
                'status': 'success',
                'signs_detected': len(results),
                'signs': results
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)