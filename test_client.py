import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import asyncio
import websockets
import cv2
import json
import base64
import threading
import queue
import time
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# --- Configuration ---
SERVER_URL = "ws://localhost:8000/ws/v1/swing_analysis"
JPEG_QUALITY = 80

# --- Thread-Safe Queues for Communication ---
frame_queue = queue.Queue(maxsize=30)
client_actions = queue.Queue()
shutdown_event = threading.Event()

# --- Network Thread Functions ---

async def network_loop(frame_q, action_q, shutdown_evt):
    """The core asyncio loop that runs in a separate thread."""
    log.info("[NET] Network thread started.")
    try:
        async with websockets.connect(SERVER_URL, ping_interval=None) as websocket:
            log.info(f"[NET] ✅ Connected to {SERVER_URL}")
            is_swinging = False

            while not shutdown_evt.is_set():
                # Check for actions from the main GUI thread
                try:
                    action = action_q.get_nowait()
                    log.info(f"[NET] Sending action: {action}")
                    await websocket.send(json.dumps({"action": action}))
                    if action == 'start_swing': is_swinging = True
                    if action == 'stop_swing': is_swinging = False
                    if action == 'stop_session': break
                except queue.Empty:
                    pass

                # If swinging, send a frame
                if is_swinging:
                    try:
                        frame = frame_q.get_nowait()
                        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                        await websocket.send(buffer.tobytes())
                    except queue.Empty:
                        pass # No frame, continue

                # Listen for server messages (non-blocking)
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                    data = json.loads(msg)
                    msg_type = data.get('type')

                    if msg_type == 'swing_analysis_complete':
                        swing_data = data.get('payload', {}).get('swing_data', {})
                        log.info(f"\n--- SWING ANALYSIS ---")
                        log.info(f"  Efficiency: {swing_data.get('efficiency_score')}% | Power: {swing_data.get('power_score')}%")
                    elif msg_type == 'session_end':
                        log.info("\n--- FINAL ARTIFACTS ---")
                        artifacts = data.get('payload', {}).get('session_artifacts', {})
                        if artifacts.get('heatmap_image'):
                            with open("final_heatmap.png", "wb") as f: f.write(base64.b64decode(artifacts['heatmap_image']))
                            log.info("  ✅ Heatmap saved.")
                        if artifacts.get('session_csv'):
                            with open("final_summary.csv", "w") as f: f.write(artifacts['session_csv'])
                            log.info("  ✅ CSV summary saved.")
                        break
                except asyncio.TimeoutError:
                    pass
    except Exception as e:
        log.error(f"[NET] ❌ Network error: {e}")
    finally:
        shutdown_evt.set()
        log.info("[NET] Network thread finished.")

# --- Main Thread (GUI and Camera) ---

if __name__ == "__main__":
    # 1. Start the network thread
    network_thread = threading.Thread(
        target=lambda: asyncio.run(network_loop(frame_queue, client_actions, shutdown_event)),
        daemon=True
    )
    network_thread.start()
    
    # Give the network thread a moment to connect
    time.sleep(2)
    if not network_thread.is_alive():
        log.error("Network thread failed to start. Is the server running? Exiting.")
        exit()

    # 2. Start the camera
    log.info("[GUI] Starting webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("[GUI] Could not open webcam.")
        shutdown_event.set()
    else:
        log.info("\n--- Controls (in window) ---\n'b' - Begin Swing\n'e' - End Swing\n'q' - Quit Session\n--------------")

    # 3. Run the main GUI loop
    while cap.isOpened() and not shutdown_event.is_set():
        ret, frame = cap.read()
        if not ret:
            log.error("[GUI] Failed to grab frame.")
            break

        cv2.imshow("Webcam Feed", frame)

        # Put frame in queue for the network thread
        if not frame_queue.full():
            frame_queue.put(frame)

        key = cv2.waitKey(1)
        if key != -1:
            if key & 0xFF == ord('b'):
                log.info("[GUI] Queuing START swing action.")
                client_actions.put('start_swing')
            elif key & 0xFF == ord('e'):
                log.info("[GUI] Queuing STOP swing action.")
                client_actions.put('stop_swing')
            elif key & 0xFF == ord('q'):
                log.info("[GUI] 'q' pressed. Shutting down.")
                client_actions.put('stop_session')
                shutdown_event.set()
                break
    
    # 4. Cleanup
    log.info("Waiting for network thread to finish...")
    network_thread.join(timeout=5)
    
    cap.release()
    cv2.destroyAllWindows()
    log.info("Application finished.")