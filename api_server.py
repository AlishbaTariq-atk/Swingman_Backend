import os
import cv2
import base64
import json
import asyncio
from typing import Dict, Any, Optional
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# --- Core Imports ---
from core.enhanced_swing_tracker import EnhancedSwingTracker
from core.swing_data_manager import SwingDataManager
from core.heatmap_generator import HeatmapGenerator
from utils.json_encoder import convert_numpy_types

OUTPUT_DIR = os.getenv("SWINGMAN_OUTPUT_DIR", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Pydantic Models ---
class TrackingPayload(BaseModel): metrics: Dict[str, Any]; pose_data: Dict[str, Any]
class TrackingUpdate(BaseModel): type: str = "tracking_update"; payload: TrackingPayload

class SwingAnalysisPayload(BaseModel): swing_data: Dict[str, Any]
class SwingAnalysisResponse(BaseModel): type: str = "swing_analysis_complete"; payload: SwingAnalysisPayload

class FinalPayload(BaseModel): session_artifacts: Dict[str, Any]
class SessionEndResponse(BaseModel): type: str = "session_end"; payload: FinalPayload

# --- API Session Logic ---
class APISwingSession:
    def __init__(self, session_name="api_session"):
        log.info("[SESSION] Initializing...")
        self.tracker = EnhancedSwingTracker(enable_pose=True)
        self.data_manager = SwingDataManager(base_dir=OUTPUT_DIR)
        self.heatmap_generator = HeatmapGenerator(output_dir=OUTPUT_DIR)
        self.session_id = self.data_manager.start_new_session(session_name)
        self.last_frame = None
        log.info(f"[SESSION] Started new session: {self.session_id}")

    def process_frame(self, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            log.warning("[SESSION] Could not decode image from bytes.")
            return None
        self.last_frame = frame.copy()
        return convert_numpy_types(self.tracker.process_frame(frame))

    def start_new_swing(self):
        log.info("[SWING] Starting new swing tracking...")
        self.tracker.start_tracking_session()

    def analyze_current_swing(self) -> Optional[Dict[str, Any]]:
        log.info("[SWING] Analyzing current swing...")
        if self.last_frame is None or len(self.tracker.swing_path_points) < 2:
            log.warning("[SWING] Not enough data to analyze swing.")
            return {"error": "Not enough swing data captured."}

        self.tracker.stop_tracking_session()
        metrics = self.tracker.get_current_metrics()

        if not metrics:
            log.error("[SWING] Analysis failed to produce metrics.")
            self.tracker.clear_current_swing()
            return {"error": "Swing analysis failed."}

        swing_data = convert_numpy_types(metrics)
        path_points = [tuple(map(int, point)) for point in self.tracker.swing_path_points]
        self.data_manager.add_swing_to_session(swing_data, self.last_frame, path_points)
        
        log.info(f"[SWING] Analysis complete. Metrics: {swing_data.get('efficiency_score')}% efficiency")
        
        if swing_data.get("impact_point"):
            points = list(self.tracker.swing_path_points)
            bat_angle = 0
            if len(points) >= 2:
                p1, p2 = points[-2:]
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                bat_angle = np.degrees(np.arctan2(dy, dx))
            
            self.heatmap_generator.add_impact_point(
                point=swing_data["impact_point"],
                bat_center=tuple(map(int, points[-1])) if points else None,
                bat_angle=bat_angle,
                efficiency_score=swing_data.get("efficiency_score", 0)
            )
            log.info("[SWING] Impact point added to heatmap.")

        self.tracker.clear_current_swing()
        return swing_data

    def end_session_and_get_artifacts(self) -> FinalPayload:
        log.info("[SESSION] Ending session and generating final artifacts...")
        self.data_manager.save_current_session()
        self.data_manager.export_data(self.session_id, "csv")
        csv_path = os.path.join(self.data_manager.base_dir, self.session_id, "swings.csv")
        csv_content = ""
        try:
            with open(csv_path, 'r') as f: csv_content = f.read()
        except FileNotFoundError: log.warning(f"CSV file not found at {csv_path}")

        heatmap_img = self.heatmap_generator.generate_heatmap_image()
        if heatmap_img is None: heatmap_img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        _, buffer = cv2.imencode('.png', heatmap_img);
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        
        artifacts = {"heatmap_image": heatmap_base64, "session_csv": csv_content}
        log.info("[SESSION] Artifact generation complete.")
        return FinalPayload(session_artifacts=artifacts)

# --- FastAPI App ---
app = FastAPI(title="Swingman Streaming API")

@app.websocket("/ws/v1/swing_analysis")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = APISwingSession()
    client_host = websocket.client.host
    log.info(f"[SERVER] Connection from {client_host} accepted.")
    try:
        while True:
            data = await websocket.receive()
            if "bytes" in data:
                results = session.process_frame(data["bytes"])
                if results:
                    await websocket.send_json(TrackingUpdate(payload=TrackingPayload(metrics=results.get('metrics',{}), pose_data=results.get('pose_data',{}))).dict())
            elif "text" in data:
                message = json.loads(data["text"])
                action = message.get("action")
                log.info(f"[SERVER] Received action '{action}' from {client_host}")
                if action == "start_swing":
                    session.start_new_swing()
                elif action == "stop_swing":
                    analysis_results = await asyncio.to_thread(session.analyze_current_swing)
                    if analysis_results:
                        await websocket.send_json(SwingAnalysisResponse(payload=SwingAnalysisPayload(swing_data=analysis_results)).dict())
                elif action == "stop_session":
                    final_payload = await asyncio.to_thread(session.end_session_and_get_artifacts)
                    await websocket.send_json(SessionEndResponse(payload=final_payload).dict())
                    break
    except WebSocketDisconnect:
        log.info(f"[SERVER] Client {client_host} disconnected unexpectedly. Saving session...")
        await asyncio.to_thread(session.data_manager.save_current_session)
    finally:
        log.info(f"[SERVER] Connection for {client_host} closed.")

@app.get("/")
def read_root(): return {"status": "ok", "title": "Swingman Streaming API"}