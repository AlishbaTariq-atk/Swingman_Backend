import os
import sys
import cv2
import argparse
import base64
from datetime import datetime
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, Any

# Add the project's root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.enhanced_swing_tracker import EnhancedSwingTracker
from core.swing_data_manager import SwingDataManager
from core.heatmap_generator import HeatmapGenerator
from utils.drawing import draw_statistics, draw_logo

# --- Data Models for API Requests and Responses ---

class SessionArgs(BaseModel):
    session_name: str = f"api_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir: str = "output"

class ProcessRequest(BaseModel):
    frame: str  # Base64 encoded image string

# --- Utility Function to Sanitize Data ---

def clean_numpy_types(data: Any) -> Any:
    """
    Recursively converts NumPy types in a dictionary or list to native Python types
    to ensure they are JSON serializable.
    """
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, dict):
        return {k: clean_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_numpy_types(i) for i in data]
    else:
        return data

# --- The Core Processing Engine (Refactored from SwingmanApp) ---

class SwingProcessor:
    def __init__(self, args: SessionArgs):
        self.args = args
        self.setup_components()
        self.is_tracking = False

    def setup_components(self):
        self.tracker = EnhancedSwingTracker(enable_pose=True)
        self.data_manager = SwingDataManager(base_dir=self.args.output_dir)
        self.session_id = self.data_manager.start_new_session(self.args.session_name)
        self.heatmap_generator = HeatmapGenerator(output_dir=self.args.output_dir)
        print(f"Initialized processor for session: {self.session_id}")

    def start_tracking_session(self):
        if self.is_tracking:
            return {"status": "error", "message": "Already tracking."}
        self.tracker.start_tracking_session()
        self.is_tracking = True
        return {"status": "success", "message": "Tracking started."}

    def process_frame(self, frame: np.ndarray):
        if not self.is_tracking:
            raise ValueError("Processing attempted without an active tracking session.")
        results = self.tracker.process_frame(frame.copy())
        processed_frame = self._draw_overlays(results['frame'], results)
        return processed_frame, results['metrics']

    def stop_tracking_session(self, final_frame: np.ndarray):
        if not self.is_tracking:
            return {"status": "error", "message": "No active tracking session to stop."}
        self.is_tracking = False
        if len(self.tracker.swing_path_points) < 2:
            self.tracker.clear_current_swing()
            return {"status": "error", "message": "Not enough data for analysis."}

        self.tracker.stop_tracking_session()
        metrics = self.tracker.get_current_metrics()
        if not metrics:
            return {"status": "error", "message": "Failed to generate metrics."}

        # The clean_numpy_types function handles this conversion automatically now
        swing_data = clean_numpy_types(metrics)
        path_points = [tuple(map(int, p)) for p in self.tracker.swing_path_points]
        self.data_manager.add_swing_to_session(swing_data, final_frame, path_points)
        if swing_data.get("impact_point"):
             self.heatmap_generator.add_impact_point(
                point=swing_data["impact_point"],
                bat_center=path_points[-1] if path_points else None,
                bat_angle=0,
                efficiency_score=swing_data.get("efficiency_score", 0)
            )
        return {"status": "success", "data": swing_data}
        
    def _draw_overlays(self, frame, results):
        if results['swing_path']:
            for i in range(1, len(results['swing_path'])):
                pt1 = tuple(map(int, results['swing_path'][i-1]))
                pt2 = tuple(map(int, results['swing_path'][i]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        if results['metrics']:
             # Clean metrics before trying to display them
             display_metrics = clean_numpy_types(results['metrics'])
             draw_statistics(frame, {
                "Efficiency": f"{display_metrics.get('efficiency_score', 0)}%",
                "Power": f"{display_metrics.get('power_score', 0)}%",
            })
        return frame
        
    def export_session_data(self):
        if not self.data_manager.current_session["swings"]:
            return {"status": "error", "message": "No swings in this session to export."}
        session_dir = self.data_manager.save_current_session()
        json_path = self.data_manager.export_data(self.session_id, "json")
        csv_path = self.data_manager.export_data(self.session_id, "csv")
        if self.heatmap_generator.normalized_impacts:
            self.heatmap_generator.save_session()
        return {"status": "success", "message": f"Session exported to {session_dir}", "files": [json_path, csv_path]}

# --- FastAPI Application ---

app = FastAPI(title="Swingman Analysis API")
sessions: Dict[str, SwingProcessor] = {}

def base64_to_frame(b64_string: str) -> np.ndarray:
    try:
        img_bytes = base64.b64decode(b64_string)
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image.")
        return frame
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 image string.")

def frame_to_base64(frame: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/session/start", summary="Start a new analysis session")
def start_session(args: SessionArgs):
    processor = SwingProcessor(args)
    session_id = processor.session_id
    sessions[session_id] = processor
    return {"session_id": session_id, "message": "Session started successfully."}

@app.post("/session/{session_id}/start_tracking", summary="Start tracking a swing")
def start_tracking(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    result = sessions[session_id].start_tracking_session()
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.post("/session/{session_id}/process_frame", summary="Process a single video frame")
def process_frame(session_id: str, request: ProcessRequest):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    processor = sessions[session_id]
    if not processor.is_tracking:
        raise HTTPException(status_code=400, detail="Tracking is not active. Call /start_tracking first.")
    frame = base64_to_frame(request.frame)
    try:
        processed_frame, metrics = processor.process_frame(frame)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    encoded_frame = frame_to_base64(processed_frame)
    
    # ** THE FIX IS HERE: Clean the metrics before returning them **
    sanitized_metrics = clean_numpy_types(metrics)
    
    return {
        "processed_frame": encoded_frame,
        "metrics": sanitized_metrics
    }

@app.post("/session/{session_id}/stop_tracking", summary="Stop tracking and analyze the swing")
def stop_tracking(session_id: str, request: ProcessRequest):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    processor = sessions[session_id]
    final_frame = base64_to_frame(request.frame)
    result = processor.stop_tracking_session(final_frame)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.post("/session/{session_id}/export", summary="Export session data")
def export_data(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    result = sessions[session_id].export_session_data()
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.on_event("shutdown")
def on_shutdown():
    print("Server is shutting down. Saving all active sessions...")
    for session_id, processor in sessions.items():
        processor.export_session_data()
    print("Cleanup complete.")

if __name__ == "__main__":
    print("Starting Swingman API server...")
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)