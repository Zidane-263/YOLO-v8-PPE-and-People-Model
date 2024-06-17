# YOLO-v8-PPE-and-People-Model

# Introduction
This repository contains an implementation of YOLOv8 (You Only Look Once, Version 8) for object detection, specifically targeting the detection of Personal Protective Equipment (PPE) and persons. The model has been trained on a custom dataset tailored to recognize different types of PPE and human figures in various environments.

# Table of Contents
1.Installation
2. Dataset Preparation
3. Training the Model
4. Evaluating the Model
5. Inference
6. Results
7. Acknowledgements

#Installation
#Prerequisites
Python 3.8 or higher
PyTorch 1.7 or higher
CUDA 10.1 or higher (if using a GPU)

# Clone the Repository
git clone https://github.com/Zidane-263/YOLO-v8-PPE-Person-Model.git
cd YOLO-v8-PPE-Person-Model

# Install Required Packages
It is recommended to create a virtual environment before installing the dependencies.
pip install -r requirements.txt

# Dataset Preparation
# Collecting Data
You will need a dataset containing images with annotations for PPE and persons. The dataset should be organized as follows:

dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/

Each subdirectory in images contains the corresponding images, and the subdirectories in labels contain the annotation files in YOLO format.

# Annotation Format
Each annotation file should have the same name as the corresponding image file and be in .txt format. Each line in the annotation file represents an object and should follow the format

<class_id> <x_center> <y_center> <width> <height>
All coordinates should be normalized to [0, 1]

# Example
If an image is named 0001.jpg, its corresponding annotation file should be 0001.txt, containing lines like:

Copy code
0 0.5 0.5 0.2 0.4
1 0.3 0.3 0.1 0.1
Here, 0 and 1 are class IDs for PPE and person, respectively.

# Training the Model
# Configuration
Before training, configure the training parameters in the config.yaml file. Key parameters include:

batch_size: Number of images per batch.
epochs: Number of training epochs.
learning_rate: Initial learning rate.
data: Path to the dataset directory.
nc: Number of classes (e.g., 2 for PPE and person).
names: List of class names (e.g., ['PPE', 'Person']).

# Inference
To perform inference on new images, use the inference.py script:

# Results
Once training and evaluation are complete, you can visualize the results using the scripts provided. Example results include detection images, performance plots, and more.

# Acknowledgements
This project builds upon the YOLOv8 implementation by Ultralytics. We acknowledge their efforts in providing a robust and efficient object detection framework.
