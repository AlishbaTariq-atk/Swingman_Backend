import asyncio
import websockets
import cv2
import json
import base64
import threading
import queue
import time

# --- Configuration ---
SERVER_URL = "ws://localhost:8000/ws/v1/swing_analysis"
JPEG_QUALITY = 80

shutdown_event = threading.Event()

def network_thread_entrypoint(frame_queue: queue.Queue):
    async def main_network_loop():
        try:
            async with websockets.connect(SERVER_URL) as websocket:
                print(f"✅ Network thread connected to {SERVER_URL}")

                while not shutdown_event.is_set():
                    try:
                        frame = frame_queue.get_nowait()
                        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                        await websocket.send(buf.tobytes())
                    except queue.Empty:
                        pass
                    except websockets.exceptions.ConnectionClosed:
                        break

                    try:
                        msg = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                        data = json.loads(msg)
                        print(f"Received: {data.get('type')}")
                    except asyncio.TimeoutError:
                        pass
                    except websockets.exceptions.ConnectionClosed:
                        break

                # Final shutdown handshake
                if not websocket.closed:
                    print("Sending 'stop_session' command...")
                    await websocket.send(json.dumps({"action": "stop_session"}))
                    print("Waiting for final artifacts...")
                    try:
                        msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        data = json.loads(msg)
                        if data.get("type") == "session_end":
                            payload = data["payload"]
                            with open("final_heatmap.png", "wb") as f:
                                f.write(base64.b64decode(payload["heatmap_image"]))
                            print("✅ Heatmap saved.")
                            with open("final_summary.csv", "w") as f:
                                f.write(payload["session_csv"])
                            print("✅ CSV summary saved.")
                    except asyncio.TimeoutError:
                        print("Error: timed out waiting for final artifacts")
        except Exception as e:
            print("Network thread error:", e)
        finally:
            print("Network thread finished.")

    asyncio.run(main_network_loop())

if __name__ == "__main__":
    frame_queue = queue.Queue(maxsize=30)
    network_thread = threading.Thread(
        target=network_thread_entrypoint, args=(frame_queue,), daemon=True
    )
    network_thread.start()

    # 🚀 Warm‑up
    time.sleep(1.0)

    print("Starting webcam... Press 'q' to stop.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        shutdown_event.set()
    else:
        while not shutdown_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Error: frame read failed.")
                break

            cv2.imshow("Webcam Feed (q to stop)", frame)
            if not frame_queue.full():
                frame_queue.put(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' pressed. Shutting down...")
                shutdown_event.set()

    print("Waiting for network thread to clean up...")
    network_thread.join(timeout=6.0)

    cap.release()
    cv2.destroyAllWindows()
    print("Application finished.")
