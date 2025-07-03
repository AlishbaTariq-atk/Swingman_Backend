import asyncio
import websockets
import cv2
import json
import base64
import threading
import queue

SERVER_URL = "ws://localhost:8001/ws/v1/swing_analysis" # Ensure port is correct
JPEG_QUALITY = 80

# Use threading Events for clearer and safer cross-thread communication
shutdown_event = threading.Event()

def network_thread_entrypoint(frame_queue: queue.Queue):
    async def main_network_loop():
        try:
            async with websockets.connect(SERVER_URL) as websocket:
                print(f"✅ Network thread connected.")

                async def send_frames():
                    while not shutdown_event.is_set():
                        try:
                            frame = frame_queue.get_nowait()
                            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                            await websocket.send(buffer.tobytes())
                        except queue.Empty:
                            await asyncio.sleep(0.01)
                        except websockets.exceptions.ConnectionClosed:
                            break
                    
                    # --- REFINED SHUTDOWN LOGIC ---
                    if not websocket.closed:
                        print("Sender: Sending stop_session command...")
                        await websocket.send(json.dumps({"action": "stop_session"}))

                async def receive_messages():
                    while not shutdown_event.is_set():
                        try:
                            message_str = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                            message = json.loads(message_str)
                            print(f"Received: {message['type']}")
                            if message.get("type") == "session_end":
                                # Handle artifacts and then allow shutdown
                                payload = message.get("payload", {})
                                with open("final_heatmap.png", "wb") as f: f.write(base64.b64decode(payload['heatmap_image']))
                                print("✅ Heatmap saved.")
                                with open("final_summary.csv", "w") as f: f.write(payload['session_csv'])
                                print("✅ CSV summary saved.")
                                break # Exit receiver loop
                        except asyncio.TimeoutError:
                            continue
                        except websockets.exceptions.ConnectionClosed:
                            break
                    
                    # --- REFINED SHUTDOWN LOGIC ---
                    # Now, wait for any final messages after shutdown is signaled.
                    # This ensures we catch the session_end response.
                    while not websocket.closed:
                         try:
                             message_str = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                             message = json.loads(message_str)
                             if message.get("type") == "session_end":
                                 # ... duplicate artifact handling in case we missed it before ...
                                 payload = message.get("payload", {})
                                 with open("final_heatmap.png", "wb") as f: f.write(base64.b64decode(payload['heatmap_image']))
                                 with open("final_summary.csv", "w") as f: f.write(payload['session_csv'])
                                 print("✅ Final artifacts received and saved.")
                         except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                             break

                await asyncio.gather(send_frames(), receive_messages())
        except Exception as e:
            print(f"Network thread error: {e}")
        finally:
            print("Network thread finished.")

    asyncio.run(main_network_loop())

if __name__ == "__main__":
    frame_queue = queue.Queue(maxsize=10)
    network_thread = threading.Thread(target=network_thread_entrypoint, args=(frame_queue,), daemon=True)
    network_thread.start()

    print("Starting webcam... Press 'q' in the window to stop.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        shutdown_event.set()
    else:
        while not shutdown_event.is_set():
            ret, frame = cap.read()
            if not ret: break
            cv2.imshow("Webcam Feed (Press 'q' to stop)", frame)
            if not frame_queue.full():
                frame_queue.put(frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                print("'q' pressed. Shutting down...")
                shutdown_event.set() # Signal all threads to stop
    
    print("Waiting for network thread to finish...")
    network_thread.join(timeout=5.0) # Give it ample time to receive final files
    cap.release()
    cv2.destroyAllWindows()
    print("Application finished.")