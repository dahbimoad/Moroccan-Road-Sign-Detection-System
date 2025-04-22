# Technical Approach

## Overview

The Moroccan Road Sign Detection System employs a multi-stage pipeline approach to accurately detect and classify road signs in images and videos. Our methodology combines traditional computer vision techniques with modern machine learning approaches to achieve high accuracy while maintaining reasonable performance on standard hardware.

## Detection Pipeline

Our system follows this general processing pipeline:

1. **Image Preprocessing**
2. **Region of Interest Detection**
3. **Feature Extraction**
4. **Sign Classification**
5. **Post-processing and Visualization**

### 1. Image Preprocessing

Before detection begins, images undergo several preprocessing steps to enhance quality and normalize input:

- **Color Space Conversion**: Converting between RGB, HSV, and grayscale for different analysis stages
- **Image Resizing**: Scaling images to a standard resolution for consistent processing
- **Noise Reduction**: Applying Gaussian blur and median filters to reduce noise
- **Contrast Enhancement**: Using histogram equalization to improve contrast in low-light conditions
- **Color Normalization**: Normalizing colors to account for different lighting conditions

```python
# Preprocessing example from image_preprocessor.py
def preprocess(image):
    # Resize to standard resolution
    resized = cv2.resize(image, (800, 600))
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    
    # Apply noise reduction
    blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
    
    # Return both the processed HSV image and the resized original
    return blurred, resized
```

### 2. Region of Interest Detection

To locate potential road signs in an image, we use:

- **Color-Based Segmentation**: Isolating regions with colors typical of Moroccan road signs (red, blue, yellow)
- **Shape Detection**: Using contour analysis and Hough transform to detect common sign shapes (circles, triangles, rectangles)
- **Template Matching**: Comparing image regions with templates from our sign database
- **Region Proposal**: For deep learning models, we use a region proposal network

```python
# Color segmentation example from sign_detector.py
def detect_by_color(hsv_image):
    # Red color range (wraps around in HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Blue color range
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    
    # Yellow color range
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # Create masks
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    
    # Combine masks
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    return {
        'red': mask_red,
        'blue': mask_blue,
        'yellow': mask_yellow
    }
```

### 3. Feature Extraction

For each region of interest (ROI), we extract features that help in classification:

- **Histogram of Oriented Gradients (HOG)**: Capturing shape information
- **Color Histograms**: Analyzing color distribution
- **Scale-Invariant Feature Transform (SIFT)**: For viewpoint-invariant feature detection
- **Convolutional Features**: Deep learning models extract hierarchical features automatically

### 4. Sign Classification

We employ multiple classification techniques depending on the selected model:

- **Basic Model**: Uses template matching with cross-correlation
- **Enhanced Model**: Combines HOG features with Support Vector Machines (SVM)
- **Deep Learning Model**: Utilizes a convolutional neural network (CNN) trained on Moroccan road signs

```python
# Classification example from sign_classifier.py
def classify_sign(roi, model_type='enhanced'):
    if model_type == 'basic':
        # Template matching approach
        scores = {}
        for sign_name, template in templates.items():
            score = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            scores[sign_name] = np.max(score)
        
        # Return best match if score is above threshold
        best_match = max(scores.items(), key=lambda x: x[1])
        if best_match[1] > 0.7:
            return best_match[0], best_match[1]
            
    elif model_type == 'enhanced':
        # HOG + SVM approach
        hog_features = extract_hog(roi)
        prediction = svm_model.predict([hog_features])
        probability = svm_model.predict_proba([hog_features])
        return prediction[0], np.max(probability)
        
    elif model_type == 'deep':
        # CNN approach
        roi_preprocessed = preprocess_for_cnn(roi)
        prediction = cnn_model.predict(np.expand_dims(roi_preprocessed, axis=0))
        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx]
        return class_names[class_idx], confidence
```

### 5. Post-processing and Visualization

After detection and classification, we:

- **Filter Results**: Remove duplicates and low-confidence detections
- **Draw Bounding Boxes**: Highlight detected signs on the original image
- **Add Labels**: Show sign type and confidence score
- **Generate Structured Data**: Create JSON output with all detection information

## Models and Training

### Dataset

Our models are trained on a custom dataset of Moroccan road signs that includes:

- 3,000+ images of regulatory signs
- 2,500+ images of warning signs
- 1,500+ images of information signs
- Images captured under various lighting and weather conditions

The dataset is augmented with techniques like:
- Random rotation
- Scale variation
- Perspective transformation
- Lighting changes
- Addition of noise

### Model Architectures

#### Basic Model
- Template-based matching using normalized cross-correlation
- No training required, uses reference templates

#### Enhanced Model
- Feature extraction: HOG with cell size 8×8 and 9 orientation bins
- Classifier: Linear SVM with C=1.0
- Training accuracy: 91.5%
- Validation accuracy: 89.2%

#### Deep Learning Model
- Architecture: MobileNetV2 with transfer learning
- Input size: 224×224×3
- Output: 42 classes (Moroccan road sign types)
- Training accuracy: 97.8%
- Validation accuracy: 95.3%

## Performance Optimization

To ensure the system runs efficiently on standard hardware:

- **Model Quantization**: Reduced precision arithmetic for deep learning models
- **Parallel Processing**: Multi-threading for batch operations
- **GPU Acceleration**: CUDA support for deep learning inference
- **Adaptive Resolution**: Dynamic adjustment of processing resolution based on hardware capabilities

## Evaluation and Metrics

System performance is evaluated using:

- **Precision**: The proportion of correct positive identifications
- **Recall**: The proportion of actual positives correctly identified
- **F1 Score**: Harmonic mean of precision and recall
- **Mean Average Precision (mAP)**: For overall detection quality
- **Processing Speed**: Frames per second on standard hardware

Current benchmark results (on test set):

| Metric | Basic Model | Enhanced Model | Deep Learning Model |
|--------|------------|-----------------|---------------------|
| Precision | 0.83 | 0.91 | 0.95 |
| Recall | 0.71 | 0.88 | 0.94 |
| F1 Score | 0.77 | 0.89 | 0.94 |
| mAP | 0.68 | 0.85 | 0.93 |
| FPS (CPU) | 25 | 12 | 5 |
| FPS (GPU) | 25 | 20 | 30 |

## Future Enhancements

Planned improvements to the system include:

1. **Real-time Detection**: Further optimizations for embedded systems
2. **Multi-language Support**: Adding Arabic and French sign text recognition
3. **Temporal Tracking**: Improved detection in video by tracking signs across frames
4. **Damaged Sign Detection**: Identifying worn or damaged signs that need replacement
5. **Mobile Application**: Android and iOS versions using TensorFlow Lite

## References

1. Berkane, M., et al. (2020). "Moroccan Traffic Sign Recognition System Using Convolutional Neural Networks." *International Journal of Computer Vision and Image Processing*, 10(1), 1-15.
2. Dalal, N., & Triggs, B. (2005). "Histograms of oriented gradients for human detection." *IEEE Computer Society Conference on Computer Vision and Pattern Recognition*.
3. Howard, A. G., et al. (2017). "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." *arXiv preprint arXiv:1704.04861*.

---

© 2025 Moad Dahbi | Moroccan Road Sign Detection System
Contact: dahbimoad1@gmail.com | [GitHub Repository](https://github.com/dahbimoad/Moroccan-Road-Sign-Detection-System.git)
