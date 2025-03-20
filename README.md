# Computer Vision for Smart Interaction

This project implements a real-time object detection system using YOLOv8 and OpenCV. It can detect and recognize real-world objects using a webcam and provide AI-driven interactions based on the detected objects.

## Features

- Train a custom YOLO model on the COCO dataset
- Real-time object detection using webcam feeds
- Object-triggered AI interactions
- Optimized for speed and efficiency
- User-friendly interface with Streamlit

## Project Structure
computer-vision-smart-interaction/
├── main.py              # Streamlit web interface
├── train.py             # Model training module
├── detect.py            # Real-time detection module
├── models/              # Trained models
├── data/                # Dataset and class names
│   └── coco_classes.json
├── datasets/            # Downloaded datasets
│   └── coco/
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
-Above, we can also download datasets, train our models, and store them.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/likhilsamala/computer-vision-smart-interaction
cd computer-vision-smart-interaction

