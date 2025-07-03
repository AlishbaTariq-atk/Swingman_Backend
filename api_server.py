import os
import cv2
import base64
import json
import asyncio
from typing import Dict, Any
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from core.enhanced_swing_tracker import EnhancedSwingTracker
from core.swing_data_manager import SwingDataManager
from core.heatmap_generator import HeatmapGenerator
from utils.json_encoder import convert_numpy_types

# --- Configuration ---
OUTPUT_DIR = os.getenv("SWINGMAN_OUTPUT_DIR", "output")

# --- Pydantic Models (Unchanged) ---
class TrackingPayload(BaseModel):
    metrics: Dict[str, Any]; pose_data: Dict[str, Any]
class TrackingUpdate(BaseModel):
    type: str = "tracking_update"; payload: TrackingPayload
class SwingAnalysisResult(BaseModel):
    type: str = "swing_analysis"; payload: Dict[str, Any]
class FinalArtifacts(BaseModel):
    heatmap_image: str; session_csv: str
class SessionEndResponse(BaseModel):
    type: str = "session_end"; payload: FinalArtifacts

class APISwingSession:
    # ... (No changes inside this class) ...
    def __init__(self, session_name="api_session"):
        print("Initializing new API swing session..."); self.tracker = EnhancedSwingTracker(enable_pose=True); self.data_manager = SwingDataManager(base_dir=OUTPUT_DIR); self.heatmap_generator = HeatmapGenerator(output_dir=OUTPUT_DIR); self.session_id = self.data_manager.start_new_session(session_name); self.last_frame = None; print(f"Started new session: {self.session_id}")
    def process_frame(self, image_bytes: bytes) -> Dict[str, Any]:
        nparr = np.frombuffer(image_bytes, np.uint8); frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR);
        if frame is None: return None
        self.last_frame = frame.copy(); results = self.tracker.process_frame(frame); return convert_numpy_types(results)
    def start_new_swing(self): self.tracker.start_tracking_session()
    def finalize_current_swing(self) -> Dict[str, Any]:
        if self.last_frame is None or len(self.tracker.swing_path_points) < 2: return {"error": "Not enough swing data captured."}
        self.tracker.stop_tracking_session(); metrics = self.tracker.get_current_metrics()
        if not metrics: return {"error": "Swing analysis failed to produce metrics."}
        swing_data = convert_numpy_types(metrics); path_points = [tuple(map(int, point)) for point in self.tracker.swing_path_points]; self.data_manager.add_swing_to_session(swing_data, self.last_frame, path_points)
        if swing_data.get("impact_point"): self.heatmap_generator.add_impact_point(point=swing_data["impact_point"])
        self.tracker.clear_current_swing(); return swing_data
    def end_session_and_get_artifacts(self) -> FinalArtifacts:
        self.data_manager.save_current_session(); self.data_manager.export_data(self.session_id, "csv")
        csv_path = os.path.join(self.data_manager.base_dir, self.session_id, "swings.csv"); csv_content = ""
        try:
            with open(csv_path, 'r') as f: csv_content = f.read()
        except FileNotFoundError: print(f"Warning: CSV file not found at {csv_path}")
        heatmap_img = self.heatmap_generator.generate_heatmap_image()
        if heatmap_img is None: heatmap_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.png', heatmap_img); heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        return FinalArtifacts(heatmap_image=heatmap_base64, session_csv=csv_content)

app = FastAPI(title="Swingman Streaming API")

@app.websocket("/ws/v1/swing_analysis")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = APISwingSession()
    try:
        while True:
            data = await websocket.receive()
            if "bytes" in data:
                results = session.process_frame(data["bytes"])
                if results:
                    payload = TrackingPayload(metrics=results.get('metrics', {}), pose_data=results.get('pose_data', {}))
                    await websocket.send_json(TrackingUpdate(payload=payload).dict())
            elif "text" in data:
                message = json.loads(data["text"])
                action = message.get("action")
                if action == "start_swing": session.start_new_swing()
                elif action == "stop_swing":
                    results = session.finalize_current_swing()
                    await websocket.send_json(SwingAnalysisResult(payload=results).dict())
                elif action == "stop_session":
                    artifacts = await asyncio.to_thread(session.end_session_and_get_artifacts)
                    await websocket.send_json(SessionEndResponse(payload=artifacts).dict())
                    break
    except WebSocketDisconnect:
        # --- REFINED EXCEPTION HANDLING ---
        print("Client disconnected unexpectedly. Cleaning up session...")
        # Still run the cleanup, but don't try to send anything back.
        await asyncio.to_thread(session.end_session_and_get_artifacts)
    finally:
        print("Closing server-side connection.")
        if not websocket.client_state == websocket.WebSocketState.DISCONNECTED:
            await websocket.close()
