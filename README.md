# Moroccan Road Sign Detection System

This project aims to detect and interpret Moroccan road signs in real-time using computer vision and image processing techniques.

## Project Overview

The system will:
1. Capture video input from a camera
2. Process frames in real-time
3. Detect road signs using image processing techniques
4. Classify and interpret the meaning of detected signs
5. Display the results in a user interface

## Project Structure

```
traitement-image/
├── data/                  # Training data and sign templates
├── docs/                  # Documentation
├── src/                   # Source code
│   ├── preprocessing/     # Image preprocessing functions
│   ├── detection/         # Sign detection algorithms
│   ├── recognition/       # Sign recognition/classification
│   ├── ui/                # User interface
│   └── main.py            # Main application entry point

```

## Getting Started

See the [installation guide](docs/installation.md) and [usage guide](docs/usage.md) to get started.

## Installation

### Requirements

- Python 3.7+
- OpenCV 4.5+
- NumPy
- Additional dependencies listed in `requirements.txt`

To install dependencies:

```bash
pip install -r requirements.txt
```

### Setup for Specific Environments

For GPU acceleration (optional):
```bash
pip install opencv-contrib-python-gpu
```

## Usage

### Running the Application

The system can be run in several modes:

```bash
# Real-time detection using webcam
python src/main.py --mode real-time

# Process a video file
python src/main.py --mode video --input path/to/video.mp4

# Process a single image
python src/main.py --mode image --input path/to/image.jpg

# Process multiple images in a directory
python src/main.py --mode batch --input path/to/image/directory
```

### Command Line Options

- `--mode`: Processing mode (`real-time`, `video`, `image`, `batch`)
- `--input`: Path to input video, image file, or directory
- `--show-steps`: Display intermediate processing steps
- `--save-output`: Save processed frames to output directory
- `--output-dir`: Directory to save output frames (default: `output`)
- `--debug`: Enable debug visualization
- `--detection-threshold`: Confidence threshold for sign detection (default: 0.5)
- `--generate-report`: Generate detection report
- `--environment`: Specify environment type (`auto`, `urban`, `desert`, `normal`)
- `--extract-templates`: Extract detected signs as templates
- `--gpu`: Use GPU acceleration if available

### Controls

During real-time or video processing:
- Press `q` to quit
- Press `s` to save the current frame
- Press `p` to pause/resume (video mode only)
- Press `r` to generate a detection report (if `--generate-report` is enabled)

## Features

### Preprocessing
- Adaptive image enhancement based on environmental conditions
- Shadow removal for urban scenes
- Brightness correction for desert conditions
- Noise reduction with edge preservation

### Detection
- Color-based segmentation tuned for Moroccan road signs
- Shape detection using contour analysis
- Multi-scale detection for varying distances

### Recognition
- Template matching for standard signs
- Feature-based classification
- Dynamic confidence scoring

### Visualization
- Real-time annotation of detected signs
- Detailed information display (sign type, meaning, confidence)
- Optional debug visualization
