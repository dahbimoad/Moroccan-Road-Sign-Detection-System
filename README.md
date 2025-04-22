# Moroccan Road Sign Detection System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-v4.5+-green.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-v5.15+-orange.svg)
![Flask](https://img.shields.io/badge/Flask-v3.1.0+-lightgrey.svg)

A comprehensive computer vision application for detecting and classifying Moroccan road signs from images and videos. This system features multiple interfaces: a command-line tool, a desktop GUI application, and a web application.

## 🌟 Features

- **Accurate Detection**: Identifies road signs in images/videos using advanced computer vision algorithms
- **Sign Classification**: Categorizes detected signs into regulatory, warning, and information signs
- **Multiple Interfaces**:
  - Command-line interface for batch processing
  - Desktop application with real-time detection
  - Web application for easy accessibility
- **Report Generation**: Creates detailed PDF reports with detection results
- **Template Management**: Add and manage your own sign templates
- **Batch Processing**: Process multiple images at once

## 📋 Requirements

- Python 3.8+
- OpenCV 4.5+
- NumPy
- PyQt5 (for desktop application)
- Flask (for web application)
- Additional dependencies in requirements.txt

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/dahbimoad/Moroccan-Road-Sign-Detection-System.git
cd Moroccan-Road-Sign-Detection-System

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Using the Command-line Interface

```bash
# Process a single image
python src/main.py --input path/to/image.jpg --output path/to/output

# Process multiple images
python src/main.py --input path/to/folder --output path/to/output --batch

# Process video
python src/main.py --input path/to/video.mp4 --output path/to/output --video
```

### Using the Desktop Application

```bash
# Launch the desktop app
python src/ui/app.py
```

### Using the Web Application

```bash
# Start the web server
python src/web/app.py
```
Then open your browser and navigate to http://localhost:5000

## 📱 Interfaces

### Command-Line Interface
Ideal for batch processing and integration into other workflows.

### Desktop Application
Features a user-friendly interface with:
- Real-time webcam detection
- Image upload and processing
- Template management
- Settings configuration
- Detailed detection reports

### Web Application
Accessible from any browser with:
- Simple image upload
- Detection result visualization
- PDF report generation
- Mobile-friendly responsive design

## 📚 Documentation

Detailed documentation is available in the `docs/` folder:
- [Installation Guide](docs/installation.md)
- [Usage Instructions](docs/usage.md)
- [Technical Approach](docs/approach.md)
- [Moroccan Road Signs Reference](docs/moroccan-signs.md)

## 🔍 How It Works

1. **Preprocessing**: Image normalization, color enhancement, and noise reduction
2. **Detection**: Identifying potential road sign regions using shape and color-based methods
3. **Feature Extraction**: Obtaining relevant visual features from detected regions
4. **Classification**: Determining the type and meaning of each detected sign

## 📸 Examples

Example results can be found in the `output/` directory after running the application.

## 👨‍💻 Author

**Moad Dahbi**
- Email: dahbimoad1@gmail.com
- GitHub: [@dahbimoad](https://github.com/dahbimoad)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenCV community for computer vision algorithms
- PyQt5 for the desktop interface
- Flask for the web application framework

---

© 2025 Moroccan Road Sign Detection System | [GitHub Repository](https://github.com/dahbimoad/Moroccan-Road-Sign-Detection-System.git)
