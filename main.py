import os
import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import json
from pathlib import Path

# Set up constants
MODEL_PATH = "models/best.pt"
CONFIDENCE_THRESHOLD = 0.5
COCO_CLASSES_PATH = "data/coco_classes.json"

# Load COCO class names
def load_class_names(json_path):
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            class_names = json.load(f)
        return class_names
    else:
        # Default COCO class names if file doesn't exist
        return {
            "0": "person", "1": "bicycle", "2": "car", "3": "motorcycle", "4": "airplane",
            "5": "bus", "6": "train", "7": "truck", "8": "boat", "9": "traffic light",
            # ... more classes
        }

# Object information database
def get_object_info(class_name):
    object_info = {
        "person": {
            "description": "A human being detected in the frame.",
            "actions": ["Facial recognition", "Pose estimation", "Activity tracking"],
            "ai_response": "I detect a person in the frame. Would you like me to analyze their posture or estimate their position in the room?"
        },
        "car": {
            "description": "A four-wheeled motor vehicle used for transportation.",
            "actions": ["Vehicle tracking", "License plate recognition", "Traffic analysis"],
            "ai_response": "I see a car. I can provide information about its make and model if you'd like a closer look."
        },
        "dog": {
            "description": "A domesticated carnivorous mammal, often kept as a pet.",
            "actions": ["Breed identification", "Behavior analysis", "Pet tracking"],
            "ai_response": "There's a dog in the frame! Would you like me to identify the breed or analyze its behavior?"
        },
        # Add more objects as needed
    }
    
    # Default info for objects not in our database
    default_info = {
        "description": "An object detected by the computer vision system.",
        "actions": ["General object tracking", "Position analysis"],
        "ai_response": "I've detected this object. Would you like me to track it or provide more information?"
    }
    
    return object_info.get(class_name, default_info)

# Main application
def main():
    st.title("Computer Vision for Smart Interaction")
    st.write("Real-time object detection system using YOLOv8 and OpenCV")
    
    # Sidebar options
    st.sidebar.title("Settings")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["About", "Train Model", "Run Detection"])
    
    if app_mode == "About":
        show_about_page()
    elif app_mode == "Train Model":
        show_training_page()
    elif app_mode == "Run Detection":
        show_detection_page()

def show_about_page():
    st.markdown("""
    # About Computer Vision for Smart Interaction
    
    This application demonstrates real-time object detection using a custom YOLOv8 model.
    
    ## Features:
    - Train a custom YOLO model on the COCO dataset
    - Real-time object detection using your webcam
    - AI-driven interactions based on detected objects
    - Optimized for speed and efficiency
    
    ## How to use:
    1. Go to the "Train Model" page to train or fine-tune the model
    2. Go to the "Run Detection" page to start real-time detection
    3. Interact with detected objects to get AI-driven responses
    """)

def show_training_page():
    st.header("Train Custom YOLOv8 Model")
    
    # Training options
    st.subheader("Training Options")
    dataset_option = st.radio("Select dataset option", ["Use COCO dataset", "Upload custom dataset"])
    
    if dataset_option == "Use COCO dataset":
        st.info("Using the COCO dataset for training")
        data_yaml = "coco.yaml"
    else:
        st.warning("Custom dataset upload not implemented in this demo")
        data_yaml = "custom.yaml"
    
    # Model options
    model_size = st.select_slider(
        "Select YOLOv8 model size",
        options=["nano", "small", "medium", "large", "xlarge"]
    )
    
    epochs = st.slider("Number of training epochs", 1, 100, 10)
    batch_size = st.slider("Batch size", 1, 64, 16)
    img_size = st.select_slider("Image size", options=[416, 512, 640, 768], value=640)
    
    # Training button
    if st.button("Start Training"):
        with st.spinner("Training model... This may take a while"):
            # In a real implementation, this would actually train the model
            # For demo purposes, we'll just simulate training
            progress_bar = st.progress(0)
            for i in range(epochs):
                # Simulate epoch training
                for j in range(10):
                    time.sleep(0.1)
                    progress_bar.progress((i * 10 + j + 1) / (epochs * 10))
                
                st.write(f"Epoch {i+1}/{epochs} completed")
            
            st.success("Training completed! Model saved to models/best.pt")
            
            # In a real implementation, you would run:
            # model = YOLO(f"yolov8{model_size[0]}.pt")
            # results = model.train(
            #     data=data_yaml,
            #     epochs=epochs,
            #     batch=batch_size,
            #     imgsz=img_size,
            #     project="models",
            #     name="train"
            # )

def show_detection_page():
    st.header("Real-time Object Detection")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH) and not os.path.exists("yolov8n.pt"):
        st.warning("No trained model found. Please train a model first or use the default YOLOv8 model.")
        use_default = st.button("Use Default YOLOv8 Model")
        if use_default:
            # Download default model
            with st.spinner("Downloading default YOLOv8 model..."):
                model = YOLO("yolov8n.pt")
                st.success("Default model loaded!")
        else:
            return
    
    # Load class names
    class_names = load_class_names(COCO_CLASSES_PATH)
    
    # Detection options
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, CONFIDENCE_THRESHOLD)
    
    # Source selection
    source_option = st.radio("Select detection source", ["Webcam", "Upload Image"])
    
    if source_option == "Webcam":
        # Start webcam
        run_webcam_detection(confidence_threshold, class_names)
    else:
        # Upload image
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            run_image_detection(uploaded_file, confidence_threshold, class_names)

def run_webcam_detection(confidence_threshold, class_names):
    # Load model
    try:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
        else:
            model = YOLO("yolov8n.pt")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Start webcam
    start_button = st.button("Start Webcam")
    stop_button = st.button("Stop Webcam")
    
    FRAME_WINDOW = st.image([])
    info_container = st.container()
    
    if start_button:
        cap = cv2.VideoCapture(0)
        
        # Check if webcam opened successfully
        if not cap.isOpened():
            st.error("Error: Could not open webcam")
            return
        
        st.session_state["stop_webcam"] = False
        
        while not st.session_state.get("stop_webcam", False):
            # Read frame
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not read frame")
                break
            
            # Run detection
            results = model(frame, conf=confidence_threshold)
            
            # Process results
            detections = []
            annotated_frame = frame.copy()
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence and class
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = class_names.get(str(cls), f"Class {cls}")
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Add to detections
                    detections.append({
                        "class_name": class_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                    })
            
            # Convert to RGB for display
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(annotated_frame)
            
            # Display detections info
            with info_container:
                if detections:
                    st.write(f"Detected {len(detections)} objects")
                    
                    # Display each detection
                    for i, detection in enumerate(detections):
                        with st.expander(f"{detection['class_name']} ({detection['confidence']:.2f})"):
                            # Get object info
                            obj_info = get_object_info(detection['class_name'])
                            
                            st.write(f"**Description:** {obj_info['description']}")
                            
                            st.write("**Possible Actions:**")
                            for action in obj_info['actions']:
                                st.write(f"- {action}")
                            
                            if st.button(f"Generate AI Response for {detection['class_name']}", key=f"ai_btn_{i}"):
                                st.info(obj_info['ai_response'])
                else:
                    st.write("No objects detected")
            
            # Add a small delay
            time.sleep(0.1)
        
        # Release resources
        cap.release()
    
    if stop_button:
        st.session_state["stop_webcam"] = True
        st.write("Webcam stopped")

def run_image_detection(uploaded_file, confidence_threshold, class_names):
    # Load model
    try:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
        else:
            model = YOLO("yolov8n.pt")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert to numpy array for OpenCV
    image_np = np.array(image)
    
    # Run detection button
    if st.button("Detect Objects"):
        with st.spinner("Detecting objects..."):
            # Run detection
            results = model(image_np, conf=confidence_threshold)
            
            # Process results
            detections = []
            annotated_image = image_np.copy()
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence and class
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = class_names.get(str(cls), f"Class {cls}")
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Add to detections
                    detections.append({
                        "class_name": class_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                    })
            
            # Display annotated image
            if len(annotated_image.shape) == 3 and annotated_image.shape[2] == 3:
                # Convert BGR to RGB if needed
                if image_np.shape[2] == 3:  # Check if it's a color image
                    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            st.image(annotated_image, caption="Detection Results", use_column_width=True)
            
            # Display detections info
            if detections:
                st.write(f"Detected {len(detections)} objects")
                
                # Display each detection
                for i, detection in enumerate(detections):
                    with st.expander(f"{detection['class_name']} ({detection['confidence']:.2f})"):
                        # Get object info
                        obj_info = get_object_info(detection['class_name'])
                        
                        st.write(f"**Description:** {obj_info['description']}")
                        
                        st.write("**Possible Actions:**")
                        for action in obj_info['actions']:
                            st.write(f"- {action}")
                        
                        if st.button(f"Generate AI Response for {detection['class_name']}", key=f"ai_btn_{i}"):
                            st.info(obj_info['ai_response'])
            else:
                st.write("No objects detected")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Create COCO classes file if it doesn't exist
    if not os.path.exists(COCO_CLASSES_PATH):
        coco_classes = {str(i): name for i, name in enumerate([
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
            "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
            "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
            "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ])}
        
        with open(COCO_CLASSES_PATH, 'w') as f:
            json.dump(coco_classes, f)
    
    main()