import cv2
import torch
import numpy as np
import time
import argparse
from ultralytics import YOLO
import json
import os

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
        # Add more objects as needed
    }
    
    # Default info for objects not in our database
    default_info = {
        "description": "An object detected by the computer vision system.",
        "actions": ["General object tracking", "Position analysis"],
        "ai_response": "I've detected this object. Would you like me to track it or provide more information?"
    }
    
    return object_info.get(class_name, default_info)

def run_detection(
    model_path,
    source=0,  # 0 for webcam, or path to video/image
    conf_threshold=0.5,
    class_names_path="data/coco_classes.json",
    show_info=True
):
    """
    Run real-time object detection using YOLOv8.
    
    Args:
        model_path: Path to the YOLOv8 model
        source: Source for detection (0 for webcam, or path to video/image)
        conf_threshold: Confidence threshold for detections
        class_names_path: Path to JSON file with class names
        show_info: Whether to show object information
    """
    # Load model
    try:
        model = YOLO(model_path)
        print(f"Model {model_path} loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load class names
    class_names = load_class_names(class_names_path)
    
    # Open video capture
    if isinstance(source, int) or source.isdigit():
        source = int(source)
        print(f"Opening webcam {source}")
    else:
        print(f"Opening video/image: {source}")
    
    cap = cv2.VideoCapture(source)
    
    # Check if opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video properties: {frame_width}x{frame_height} @ {fps}fps")
    
    # Create window
    cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8 Detection", frame_width, frame_height)
    
    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    
    # Selected object for info display
    selected_object = None
    
    print("Press 'q' to quit, 'i' to toggle info display")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("End of video stream")
            break
        
        # Increment frame count
        frame_count += 1
        
        # Calculate FPS every 10 frames
        if frame_count % 10 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps_display = 10 / elapsed_time
            start_time = end_time
        
        # Run detection
        results = model(frame, conf=conf_threshold)
        
        # Process results
        detections = []
        
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
                color = (0, 255, 0)  # Green
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Add to detections
                detections.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display number of detections
        cv2.putText(frame, f"Objects: {len(detections)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display object info if enabled
        if show_info and selected_object is not None:
            # Find the selected object in current detections
            selected_detection = None
            for detection in detections:
                if detection["class_name"] == selected_object:
                    selected_detection = detection
                    break
            
            if selected_detection:
                # Get object info
                obj_info = get_object_info(selected_object)
                
                # Create info panel
                info_panel = np.zeros((frame_height, 300, 3), dtype=np.uint8)
                
                # Add title
                cv2.putText(info_panel, selected_object, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add description
                cv2.putText(info_panel, "Description:", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                # Split description into multiple lines
                desc = obj_info["description"]
                words = desc.split()
                lines = []
                current_line = ""
                
                for word in words:
                    if len(current_line + " " + word) <= 40:
                        current_line += " " + word if current_line else word
                    else:
                        lines.append(current_line)
                        current_line = word
                
                if current_line:
                    lines.append(current_line)
                
                # Draw description lines
                y_pos = 90
                for line in lines:
                    cv2.putText(info_panel, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_pos += 25
                
                # Add actions
                cv2.putText(info_panel, "Possible Actions:", (10, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                y_pos += 50
                for action in obj_info["actions"]:
                    cv2.putText(info_panel, f"- {action}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_pos += 25
                
                # Add AI response
                cv2.putText(info_panel, "AI Response:", (10, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                # Split AI response into multiple lines
                ai_resp = obj_info["ai_response"]
                words = ai_resp.split()
                lines = []
                current_line = ""
                
                for word in words:
                    if len(current_line + " " + word) <= 40:
                        current_line += " " + word if current_line else word
                    else:
                        lines.append(current_line)
                        current_line = word
                
                if current_line:
                    lines.append(current_line)
                
                # Draw AI response lines
                y_pos += 50
                for line in lines:
                    cv2.putText(info_panel, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_pos += 25
                
                # Combine frame and info panel
                combined_frame = np.hstack((frame, info_panel))
                cv2.imshow("YOLOv8 Detection", combined_frame)
            else:
                # Object no longer detected
                selected_object = None
                cv2.imshow("YOLOv8 Detection", frame)
        else:
            cv2.imshow("YOLOv8 Detection", frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' to quit
        if key == ord('q'):
            break
        
        # 'i' to toggle info display
        elif key == ord('i'):
            show_info = not show_info
            print(f"Info display {'enabled' if show_info else 'disabled'}")
        
        # Number keys to select object
        elif key >= ord('1') and key <= ord('9'):
            idx = key - ord('1')
            if idx < len(detections):
                selected_object = detections[idx]["class_name"]
                print(f"Selected object: {selected_object}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Run real-time object detection using YOLOv8")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLOv8 model")
    parser.add_argument("--source", type=str, default="0", help="Source for detection (0 for webcam, or path to video/image)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detections")
    parser.add_argument("--class-names", type=str, default="data/coco_classes.json", help="Path to JSON file with class names")
    parser.add_argument("--no-info", action="store_true", help="Disable object information display")
    
    args = parser.parse_args()
    
    # Run detection
    run_detection(
        model_path=args.model,
        source=args.source,
        conf_threshold=args.conf,
        class_names_path=args.class_names,
        show_info=not args.no_info
    )

if __name__ == "__main__":
    main()