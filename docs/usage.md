# Usage Instructions

The Moroccan Road Sign Detection System offers three different interfaces to accommodate various use cases:
1. Command-Line Interface (CLI) for batch processing and integration
2. Desktop Application for interactive use with a graphical interface
3. Web Application for easy access without installation

## Command-Line Interface

The CLI is ideal for batch processing, automation, and integration with other systems.

### Basic Usage

```bash
python src/main.py --input <input_path> --output <output_directory>
```

### Available Options

- `--input` or `-i`: Path to input image, video, or directory (required)
- `--output` or `-o`: Path to output directory (required)
- `--batch` or `-b`: Process all images in the input directory
- `--video` or `-v`: Process video input
- `--webcam` or `-w`: Use webcam as input source
- `--confidence` or `-c`: Confidence threshold (0.0-1.0, default: 0.5)
- `--model` or `-m`: Choose model type ('basic', 'enhanced', 'deep', default: 'enhanced')
- `--report` or `-r`: Generate PDF report
- `--visualize` or `-z`: Show visualization window during processing
- `--help` or `-h`: Show help message

### Examples

```bash
# Process a single image
python src/main.py --input images/road.jpg --output results/

# Process all images in a directory
python src/main.py --input images/ --output results/ --batch

# Process a video file
python src/main.py --input videos/driving.mp4 --output results/ --video

# Use webcam input
python src/main.py --output results/ --webcam

# Generate a PDF report
python src/main.py --input images/road.jpg --output results/ --report
```

### Batch Processing Example

```bash
# Process multiple images with enhanced model and generate reports
python src/main.py --input data/test_images/ --output results/ --batch --model enhanced --report
```

## Desktop Application

The desktop application offers a user-friendly GUI for interactive road sign detection.

### Starting the Application

```bash
python src/ui/app.py
```

### Main Window Overview

The desktop application features:

1. **Main Toolbar**
   - Open Image: Load an image from your computer
   - Open Folder: Batch process all images in a folder
   - Open Camera: Use webcam as input source
   - Save Results: Save current detection results
   - Settings: Configure application settings

2. **Workspace Area**
   - Left panel: Original image or video feed
   - Right panel: Processed image with detected signs
   - Bottom panel: List of detected signs with details

3. **Status Bar**
   - Processing status
   - Current model in use
   - Confidence threshold

### Step-by-Step Usage

1. **Loading an Image**
   - Click the "Open Image" button or use Ctrl+O
   - Select an image file containing road signs
   - The image will be processed automatically

2. **Using the Webcam**
   - Click the "Open Camera" button
   - Adjust settings if needed (click the "Settings" button)
   - Point your camera at road signs
   - Click "Capture" to freeze the frame and process it

3. **Batch Processing**
   - Click the "Open Folder" button
   - Select a folder containing multiple images
   - Set batch processing options in the dialog
   - Click "Process" to start batch processing

4. **Managing Templates**
   - Go to Tools → Template Management
   - Add new templates for better detection
   - Edit or remove existing templates
   - Import templates from PDF documents

5. **Configuring Settings**
   - Click the "Settings" button or go to Tools → Settings
   - Adjust detection parameters
   - Configure visualization options
   - Set output preferences

6. **Generating Reports**
   - After detection, click "Generate Report" or go to File → Generate Report
   - Choose report format and content options
   - Select output location
   - View and/or print the generated report

### Keyboard Shortcuts

- Ctrl+O: Open image
- Ctrl+F: Open folder
- Ctrl+W: Open webcam
- Ctrl+S: Save results
- Ctrl+R: Generate report
- Ctrl+T: Open template management
- Ctrl+P: Open settings
- F1: Help
- Esc: Cancel current operation

## Web Application

The web application provides easy access to road sign detection from any device with a web browser.

### Starting the Web Server

```bash
python src/web/app.py
```

Then open your web browser and navigate to http://localhost:5000

### Web Interface Features

1. **Home Page**
   - File upload form for submitting images
   - Real-time image preview before upload
   - Information about supported road sign types

2. **Results Page**
   - Side-by-side comparison of original and processed images
   - Interactive image comparison slider
   - Detailed table of all detected road signs
   - PDF report download option

3. **About Page**
   - Project information and technical details
   - Documentation on the detection process
   - API usage examples for developers

### Using the Web Interface

1. **Uploading an Image**
   - Navigate to the home page
   - Click "Choose File" or drag and drop an image
   - Click "Detect Road Signs" to process the image

2. **Viewing Results**
   - Examine the original and processed images
   - Use the image comparison slider to compare before/after
   - Review the table of detected signs with details
   - Click "Analyze Another Image" to process a new image

3. **Downloading Reports**
   - On the results page, click "Download Report"
   - A PDF document will be generated with detection results
   - The report includes images and detection details

4. **Using the API (for developers)**
   - The web application also provides a REST API
   - Send a POST request to `/api/detect` with an image file
   - Receive JSON results with detection data

## Advanced Usage

### Custom Templates

Create custom templates for specific Moroccan road signs:

1. Prepare a clear image of the sign
2. Use Tools → Template Extraction in the desktop app
3. Crop and adjust the template
4. Save the template with appropriate metadata
5. The system will use your custom templates for future detections

### Training Custom Models

For advanced users who want to train custom models:

1. Prepare a dataset of Moroccan road sign images in `data/training/`
2. Use data augmentation to expand your dataset:
   ```bash
   python src/utils/data_augmentor.py --input data/training/ --output data/augmented/
   ```
3. Train the model using your preferred method
4. Update the model configuration to use your custom model

### Integration with Other Systems

The system can be integrated with other applications:

- Use the CLI for script-based integration
- Use the REST API for web-based integration
- Import the detection modules directly into your Python code

```python
from src.detection.sign_detector import SignDetector

# Initialize the detector
detector = SignDetector()

# Process an image
results = detector.detect('path/to/image.jpg')

# Use the results
for sign in results:
    print(f"Found sign: {sign.type} with confidence {sign.confidence}")
```

## Troubleshooting

- **Poor detection quality**: Try adjusting confidence threshold in settings
- **Slow processing**: Reduce image resolution or use the 'basic' model for faster processing
- **Missing signs**: Add custom templates or adjust detection parameters
- **Web app not starting**: Check if port 5000 is already in use

For any issues, please refer to our [GitHub Issues](https://github.com/dahbimoad/Moroccan-Road-Sign-Detection-System/issues) page or contact the developer directly at dahbimoad1@gmail.com.

---

© 2025 Moad Dahbi | Moroccan Road Sign Detection System
