## Face-Recognition 
In this project, I use YOLO for face detection and resnet (Tensorflow) for face recognition.

## Description
### Face Recognition
Face Recognition is a computer vision task that involves identifying and verifying a person's identity from digital images or video frames. It is commonly used in security systems, biometric authentication, and social media applications.

### ResNet (TensorFlow)
ResNet, short for Residual Network, is a deep convolutional neural network architecture introduced by Kaiming He et al. in their paper **Deep Residual Learning for Image Recognition**.



## Table of Contents
- [System Diagram](#system-diagram)
- [Training Process and Recognition Diagram](#training-process-and-recognition-diagram)
- [Installation](#installation)
- [Author](#author)
- [License](#license)

##
## System Diagram
![System Diagram](https://raw.githubusercontent.com/WaiHninEaindrarMg/Face-Recognition/main/figure/Overview%20System.jpg)

## Training Process and Recognition Diagram
![Training Process and Recognition Diagram](https://raw.githubusercontent.com/WaiHninEaindrarMg/Face-Recognition/main/figure/TrainingProcess_Recognition.jpg)


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
After running the specified file, the script automatically stores these folders and datasets:
![Folders and Datasets](https://raw.githubusercontent.com/WaiHninEaindrarMg/Face-Recognition/main/figure/folders_dataset.jpg)


2. Run this file Train.ipynb
```
Run Train.ipynb
```
In this Train.ipynb , There are three resnet model 50, 101, 152 model.
This is performace plot for train and validation accuracy.
![Accuracy](https://raw.githubusercontent.com/WaiHninEaindrarMg/Face-Recognition/main/figure/resnet%20performance.jpg)


3. Run this file face_detect_recong_2.py
```
Run face_detect_recong_2.py
```
After run this face_detect_recong_2.py, video output will be showed.
This is result video for 3 models (face recognition results)
![Result](https://raw.githubusercontent.com/WaiHninEaindrarMg/Face-Recognition/main/figure/resnet_accuracy.gif)

##
## Author
👤 : Wai Hnin Eaindrar Mg  
📧 : [waihnineaindrarmg@gmail.com](mailto:waihnineaindrarmg@gmail.com)

## License

This project is licensed under the [MIT License](LICENSE.md) - see the LICENSE.md file for details.

