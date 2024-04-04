## Face-Recognition 
In this project, I use YOLO for face detection and resnet (Tensorflow) for face recognition.

## Description
### Face Recognition
Face Recognition is a computer vision task that involves identifying and verifying a person's identity from digital images or video frames. It is commonly used in security systems, biometric authentication, and social media applications.

### ResNet (TensorFlow)
ResNet, short for Residual Network, is a deep convolutional neural network architecture introduced by Kaiming He et al. in their paper **Deep Residual Learning for Image Recognition**.



## Table of Contents
- [System Diagram](#system-diagram)
- [Training Process and Recognition Diagram](#training-diagram)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Contributing](#contributing)
- [License](#license)

## System Diagram
![System Diagram](https://raw.githubusercontent.com/WaiHninEaindrarMg/Face-Recognition/main/Overview%20System.jpg)

## Training Process and Recognition Diagram
![Training Process and Recognition Diagram](https://raw.githubusercontent.com/WaiHninEaindrarMg/Face-Recognition/main/TrainingProcess_Recognition.png)


## Installation
1. Clone the repository:
```
git clone https://github.com/WaiHninEaindrarMg/Face-Recognition.git
```

2. Install Ultralytics , check here for more information (https://github.com/akanametov/yolov8-face) :
I used pretrained YOLO Model from (https://github.com/akanametov/yolov8-face) : 
```
pip install ultralytics
```

3. Install tensorflow:
```
pip install tensorflow==2.10.1
```

## Instruction
1. Run this file https://github.com/WaiHninEaindrarMg/Face-Recognition/blob/main/face_detection_dataset_extraction_.py
```
python face_detection_dataset_extraction_.py
```
After run this file, these folders and datasets are automatically stored.

