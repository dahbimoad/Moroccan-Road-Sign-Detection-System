# Installation Guide

## Prerequisites

- Python 3.8 or later
- Webcam or video input device (for real-time detection)
- Sufficient processing power for real-time analysis

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/dahbimoad/Moroccan-Road-Sign-Detection-System.git
   cd Moroccan-Road-Sign-Detection-System
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Required Libraries

The `requirements.txt` file includes:

```
numpy==2.2.5
opencv-python==4.11.0.86
matplotlib==3.10.1
scikit-image==0.25.2
scikit-learn==1.6.1
tensorflow==2.9.1  # Optional, if using deep learning
PyQt5==5.15.11
Flask==3.1.0
Werkzeug==3.1.3
pillow==11.2.1
pdf2image==1.17.0
pytesseract==0.3.13
```

## System-Specific Installation Notes

### Windows

1. Install Python from [python.org](https://www.python.org/downloads/)
2. You may need to install Visual C++ Build Tools for some packages:
   - Download and install from [Visual Studio Downloads](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

3. For PyQt5:
   ```
   pip install PyQt5
   ```

4. For Tesseract OCR (optional - used for text recognition in signs):
   - Download the installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - Add the Tesseract installation directory to your system PATH

### macOS

1. Install Python and pip using Homebrew:
   ```
   brew install python
   ```

2. For PyQt5:
   ```
   pip install PyQt5
   ```

3. For Tesseract OCR (optional):
   ```
   brew install tesseract
   ```

### Linux

1. Install Python:
   ```
   sudo apt-get update
   sudo apt-get install python3 python3-pip
   ```

2. For PyQt5:
   ```
   sudo apt-get install python3-pyqt5
   ```

3. For Tesseract OCR (optional):
   ```
   sudo apt-get install tesseract-ocr
   ```

## Verifying Installation

To verify that the installation was successful:

1. Activate your virtual environment if you created one:
   ```
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Run the installation verification script:
   ```
   python src/main.py --verify
   ```

   This will check if all required libraries are properly installed.

## Components Installation

### Command-line Interface

No additional installation required beyond the main requirements.

### Desktop Application

The PyQt5 package included in the requirements.txt will install everything needed for the desktop GUI application.

### Web Application

The Flask package included in the requirements.txt will install everything needed for the web application.

To create necessary directories for uploaded and processed images:

```bash
mkdir -p src/web/static/uploads
mkdir -p src/web/static/results
```

## Configuration

The default configuration should work in most environments, but you can customize the settings:

- For the command-line and desktop applications, settings are stored in `data/config.json`
- For the web application, settings are in `src/web/config.py`

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'cv2'**
   - Solution: Reinstall OpenCV: `pip install opencv-python`

2. **ImportError: No module named 'PyQt5'**
   - Solution: Install PyQt5: `pip install PyQt5`

3. **Flask application fails to start**
   - Solution: Check if port 5000 is already in use. You can change the port in `src/web/app.py`

4. **Missing model files**
   - Solution: Download the pre-trained models from the releases page of our GitHub repository

### Getting Help

If you encounter any problems during installation, please:

1. Check the [GitHub Issues](https://github.com/dahbimoad/Moroccan-Road-Sign-Detection-System/issues) for similar problems
2. Open a new issue if your problem has not been reported
3. Contact the developer directly at dahbimoad1@gmail.com

## Next Steps

After installation, check the [Usage Instructions](usage.md) to get started with the application.

---

© 2025 Moad Dahbi | Moroccan Road Sign Detection System
