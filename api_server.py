import os
import sys
import cv2
import argparse
import base64
from datetime import datetime
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from contextlib import asynccontextmanager
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Project Setup ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.enhanced_swing_tracker import EnhancedSwingTracker
from core.swing_data_manager import SwingDataManager
from core.heatmap_generator import HeatmapGenerator
from utils.drawing import draw_statistics

# --- Data Models ---
class SessionArgs(BaseModel):
    session_name: str = f"api_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir: str = "output"
class ProcessRequest(BaseModel): frame: str
class ExportedFile(BaseModel): filename: str; content_b64: str; mimetype: str
class ExportResponse(BaseModel): status: str; files: List[ExportedFile]

# --- Core Logic ---
def clean_numpy_types(data: Any) -> Any:
    if isinstance(data, np.integer): return int(data)
    if isinstance(data, np.floating): return float(data)
    if isinstance(data, np.ndarray): return data.tolist()
    if isinstance(data, np.bool_): return bool(data)
    if isinstance(data, dict): return {k: clean_numpy_types(v) for k, v in data.items()}
    if isinstance(data, list): return [clean_numpy_types(i) for i in data]
    return data

class SwingProcessor:
    def __init__(self, args: SessionArgs):
        self.args = args; self.setup_components(); self.is_tracking = False
    def setup_components(self):
        self.tracker = EnhancedSwingTracker(enable_pose=True)
        self.data_manager = SwingDataManager(base_dir=self.args.output_dir)
        self.session_id = self.data_manager.start_new_session(self.args.session_name)
        self.heatmap_generator = HeatmapGenerator(output_dir=self.args.output_dir)
        logging.info(f"Initialized processor for session: {self.session_id}")
    def start_tracking_session(self):
        self.tracker.start_tracking_session(); self.is_tracking = True
        return {"status": "success", "message": "Tracking started."}
    def process_frame(self, frame: np.ndarray):
        results = self.tracker.process_frame(frame.copy())
        return self._draw_overlays(results['frame'], results), results['metrics']
    def stop_tracking_session(self, final_frame: np.ndarray):
        if not self.is_tracking: return {"status": "error", "message": "No active tracking session."}
        self.is_tracking = False
        if len(self.tracker.swing_path_points) < 2: self.tracker.clear_current_swing(); return {"status": "error", "message": "Not enough data."}
        self.tracker.stop_tracking_session(); metrics = self.tracker.get_current_metrics()
        if not metrics: return {"status": "error", "message": "Failed to generate metrics."}
        swing_data = clean_numpy_types(metrics)
        path_points = [tuple(map(int, p)) for p in self.tracker.swing_path_points]
        self.data_manager.add_swing_to_session(swing_data, final_frame, path_points)
        if swing_data.get("impact_point"): self.heatmap_generator.add_impact_point(point=swing_data["impact_point"], bat_center=path_points[-1] if path_points else None, bat_angle=0, efficiency_score=swing_data.get("efficiency_score", 0))
        return {"status": "success", "data": swing_data}
    def _draw_overlays(self, frame, results):
        if results['swing_path']:
            for i in range(1, len(results['swing_path'])):
                pt1=tuple(map(int, results['swing_path'][i-1])); pt2=tuple(map(int, results['swing_path'][i])); cv2.line(frame, pt1, pt2, (0,255,0), 2)
        if results['metrics']:
             display_metrics = clean_numpy_types(results['metrics'])
             draw_statistics(frame, {"Efficiency": f"{display_metrics.get('efficiency_score', 0)}%", "Power": f"{display_metrics.get('power_score', 0)}%"})
        return frame

    def export_session_data(self) -> List[Dict[str, str]]:
        logging.info(f"[{self.session_id}] --- Initiating Export Process ---")
        if not self.data_manager.current_session["swings"]:
            logging.warning(f"[{self.session_id}] No swings to export."); return []
        
        exported_files = []
        session_dir = self.data_manager.save_current_session()
        
        # --- CORRECT CSV EXPORT ---
        self.data_manager.export_data(self.session_id, "csv")
        # ** THE FIX: Manually construct the correct path **
        csv_filename = "swings.csv"
        csv_path = os.path.join(session_dir, csv_filename)
        
        logging.info(f"[{self.session_id}] Checking for CSV at: {csv_path}")
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f: content = f.read()
            b64_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
            exported_files.append({"filename": csv_filename, "content_b64": b64_content, "mimetype": "text/csv"})
            logging.info(f"[{self.session_id}] Successfully processed CSV file.")
        else:
            logging.error(f"[{self.session_id}] Could not find generated CSV file at expected path.")

        # --- CORRECT HEATMAP EXPORT ---
        if self.heatmap_generator.normalized_impacts:
            heatmap_img = self.heatmap_generator.generate_heatmap_image()
            heatmap_filename = "impact_heatmap.png"
            heatmap_path = os.path.join(session_dir, heatmap_filename)
            cv2.imwrite(heatmap_path, heatmap_img)
            
            with open(heatmap_path, 'rb') as f: content = f.read()
            b64_content = base64.b64encode(content).decode('utf-8')
            exported_files.append({"filename": heatmap_filename, "content_b64": b64_content, "mimetype": "image/png"})
            logging.info(f"[{self.session_id}] Successfully processed heatmap file.")
        else:
            logging.info(f"[{self.session_id}] No impact data for heatmap. Skipping.")
            
        logging.info(f"[{self.session_id}] --- Export Finished. Returning {len(exported_files)} file(s). ---")
        return exported_files

# --- FastAPI Setup ---
sessions: Dict[str, SwingProcessor] = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("🚀 Swingman API is starting up...")
    yield
    logging.info("🔌 Server is shutting down...")

app = FastAPI(title="Swingman Analysis API", lifespan=lifespan)

# --- Endpoints ---
def base64_to_frame(b64_string: str) -> np.ndarray:
    try:
        img_bytes = base64.b64decode(b64_string); img_arr = np.frombuffer(img_bytes, dtype=np.uint8); frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if frame is None: raise ValueError("Could not decode image.")
        return frame
    except Exception: raise HTTPException(status_code=400, detail="Invalid Base64 image string.")
def frame_to_base64(frame: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', frame); return base64.b64encode(buffer).decode('utf-8')

@app.post("/session/start", tags=["Session"])
def start_session(args: SessionArgs):
    processor = SwingProcessor(args); session_id = processor.session_id; sessions[session_id] = processor
    return {"session_id": session_id, "message": "Session started successfully."}

@app.post("/session/{session_id}/start_tracking", tags=["Processing"])
def start_tracking(session_id: str):
    if session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found.")
    return sessions[session_id].start_tracking_session()

@app.post("/session/{session_id}/process_frame", tags=["Processing"])
def process_frame(session_id: str, request: ProcessRequest):
    if session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found.")
    processor = sessions[session_id];
    if not processor.is_tracking: raise HTTPException(status_code=400, detail="Tracking is not active.")
    frame = base64_to_frame(request.frame)
    processed_frame, metrics = processor.process_frame(frame)
    return {"processed_frame": frame_to_base64(processed_frame), "metrics": clean_numpy_types(metrics)}

@app.post("/session/{session_id}/stop_tracking", tags=["Processing"])
def stop_tracking(session_id: str, request: ProcessRequest):
    if session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found.")
    result = sessions[session_id].stop_tracking_session(base64_to_frame(request.frame))
    if result["status"] == "error": raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.post("/session/{session_id}/export", response_model=ExportResponse, tags=["Session"])
def export_data(session_id: str):
    if session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found.")
    files_data = sessions[session_id].export_session_data()
    # Now we can safely check if the list is empty and respond accordingly
    if not files_data: raise HTTPException(status_code=404, detail="No exportable files were generated.")
    return {"status": "success", "files": files_data}

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)