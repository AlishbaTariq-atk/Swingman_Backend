"""
test_ml_modules.py - Test both ML modules together
"""

import cv2
import sys
import os

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from pose_analyzer import PoseAnalyzer
from yolo_detector import YoloDetector

def test_both_modules():
    """Test both MediaPipe Pose and YOLOv8 together"""
    print("Initializing both ML modules...")
    
    # Initialize both analyzers
    pose_analyzer = PoseAnalyzer()
    yolo_detector = YoloDetector()
    
    # Check if both are available
    if not yolo_detector.is_available():
        print("Warning: YOLO not available, testing pose only")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Combined ML test running...")
    print("Press 'q' to quit, 'p' for pose only, 'y' for YOLO only, 'b' for both")
    
    mode = 'both'  # Default mode
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create a copy for processing
        display_frame = frame.copy()
        
        # Run pose analysis
        if mode in ['pose', 'both']:
            pose_data = pose_analyzer.analyze_pose(frame)
            display_frame = pose_analyzer.draw_pose(display_frame, pose_data)
        
        # Run YOLO detection
        if mode in ['yolo', 'both'] and yolo_detector.is_available():
            detections = yolo_detector.detect_objects(frame)
            display_frame = yolo_detector.draw_detections(display_frame, detections)
        
        # Add mode indicator
        cv2.putText(display_frame, f"Mode: {mode.upper()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Show frame
        cv2.imshow("ML Modules Test", display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            mode = 'pose'
            print("Switched to pose-only mode")
        elif key == ord('y'):
            mode = 'yolo'
            print("Switched to YOLO-only mode")
        elif key == ord('b'):
            mode = 'both'
            print("Switched to both modules mode")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Combined test completed!")

if __name__ == "__main__":
    test_both_modules()
