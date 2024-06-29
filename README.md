# Image Segmentation using Mask R-CNN and U-Net
Image segmentation is a fundamental task in computer vision that involves partitioning an image into multiple segments or regions, each of which corresponds to different objects or parts ofobjects. The primary goal of image segmentation is to simplify the representation of an image and make it more meaningful and easier to analyze.
## Description
This project demonstrates image segmentation using Mask R-CNN and U-Net. The models are applied to segment objects in various images, highlighting their applications in fields such as medical imaging and autonomous driving.
## Table of Contents
## Types of Image Segmentation
### Semantic Segmentation 
- **Definition**: Assigns a label to each pixel in the image, classifying each pixel into a predefined category. All pixels belonging to the same class are given the same label.
- **Applications**: 
  - **Autonomous Driving**: Identifying different elements such as roads, vehicles, pedestrians, and traffic signs.
  - **Medical Imaging**: Segmenting different anatomical structures like organs, tissues, and tumors.
  - **Satellite Imagery**: Classifying land cover types like forests, water bodies, and urban areas.
- **Example**: Segmenting an image of a street scene into classes such as road, cars, pedestrians, and buildings.
  
![Semantic Segmantation Example](Semantic_Segmentation.png)

### Instance Segmentation
- **Definition**: Extends semantic segmentation by distinguishing between different instances of the same class. Each object instance is segmented separately.
- **Applications**:
  - **Object Detection**: Identifying and segmenting each object in an image individually, useful in inventory management and robotics.
  - **Video Surveillance**: Tracking individual objects or people in real-time for security purposes.
  - **Medical Imaging**: Separating overlapping cells in microscopy images.
- **Example**: Segmenting each person in a crowd as a separate entity, not just as part of the 'person' class.

![Instance Segmentation Example]
  
