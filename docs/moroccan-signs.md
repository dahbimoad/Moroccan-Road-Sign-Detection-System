# Moroccan Road Signs

## Categories of Moroccan Road Signs

Moroccan road signs generally follow international standards but may have some unique designs. Here are the main categories:

### 1. Warning Signs
- Triangular shape with red border
- Yellow or white background
- Examples: dangerous curve, pedestrian crossing, intersection

### 2. Prohibition Signs
- Circular shape with red border
- White background
- Examples: no entry, no parking, speed limit

### 3. Mandatory Signs
- Circular shape
- Blue background
- Examples: go straight, turn right, minimum speed

### 4. Information Signs
- Rectangular shape
- Blue background
- Examples: parking, hospital, one-way street

## Distinctive Features for Detection

When implementing detection algorithms, focus on:

1. **Colors**: Red, blue, yellow, and white are predominant in road signs
2. **Shapes**: Triangles, circles, and rectangles are the main shapes
3. **Symbols**: The pictograms inside the signs

## Image Processing Techniques for Moroccan Signs

- **Color Segmentation**: Use HSV color space to isolate red, blue, and yellow regions
- **Shape Detection**: Apply Hough transforms to detect circles and line segments for triangles/rectangles
- **Symbol Recognition**: Use template matching or feature-based approaches

## Creating a Sign Database

To build a comprehensive system:

1. Collect images of Moroccan road signs from:
   - Manual photography in different lighting conditions
   - Online resources
   - Official traffic documentation

2. For each sign, document:
   - Visual appearance
   - Meaning/interpretation
   - Category
   - Shape and color properties
   
3. Create templates for matching algorithms

## Test Cases

Ensure your system can handle:
- Different lighting conditions (day/night)
- Partial occlusion
- Weather effects (rain, fog)
- Damaged or faded signs
- Signs at different distances and angles
