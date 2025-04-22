# Installation Guide

## Prerequisites

- Python 3.8 or later
- Webcam or video input device
- Sufficient processing power for real-time analysis

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/moroccan-road-sign-detection.git
   cd moroccan-road-sign-detection
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

The `requirements.txt` file should include:

```
numpy==1.22.3
opencv-python==4.5.5.64
matplotlib==3.5.1
scikit-image==0.19.2
scikit-learn==1.0.2
tensorflow==2.9.1  # Optional, if using deep learning
```

These libraries provide the foundation for image processing, computer vision, and machine learning that will be used in this project.
