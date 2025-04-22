# Usage Guide

## Running the Application

To run the application in real-time mode:

```
python src/main.py --mode real-time
```

To process a pre-recorded video:

```
python src/main.py --mode video --input path/to/video.mp4
```

To process a single image:

```
python src/main.py --mode image --input path/to/image.jpg
```

## User Interface

The application displays:
- Original video feed
- Processed frame with highlighted detections
- Information about detected road signs
- Confidence score of the detection/classification

## Keyboard Controls

- Press 'q' to quit the application
- Press 's' to save the current frame
- Press 'p' to pause/resume the video feed
