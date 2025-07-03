import asyncio
import websockets
import cv2
import json
import base64
import threading
import queue  # Use the standard thread-safe queue

# --- Configuration ---
SERVER_URL = "ws://localhost:8001/ws/v1/swing_analysis"
JPEG_QUALITY = 80

# A thread-safe flag to signal shutdown
running = True

def network_thread_entrypoint(frame_queue: queue.Queue):
    """
    This function is the entry point for our background network thread.
    It creates and runs its own asyncio event loop.
    """
    async def main_network_loop():
        """The main async logic for sending and receiving data."""
        global running
        try:
            async with websockets.connect(SERVER_URL) as websocket:
                print(f"✅ Network thread connected to server at {SERVER_URL}")

                # Task to continuously send frames from the queue
                async def send_frames():
                    while running:
                        try:
                            frame = frame_queue.get_nowait()
                            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                            await websocket.send(buffer.tobytes())
                            frame_queue.task_done()
                        except queue.Empty:
                            await asyncio.sleep(0.01)
                        except websockets.exceptions.ConnectionClosed:
                            break
                    
                    if not websocket.closed:
                        print("Network thread: Sending stop_session command...")
                        # This command name must match your api_server.py
                        await websocket.send(json.dumps({"action": "stop_session"}))

                # Task to continuously receive messages from the server
                async def receive_messages():
                    global running
                    while running:
                        try:
                            message_str = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                            message = json.loads(message_str)
                            msg_type = message.get("type")

                            # --- THIS IS THE ADDED LOGIC ---
                            if msg_type == "session_end":
                                print("\n--- FINAL ARTIFACTS RECEIVED ---")
                                payload = message.get("payload", {})
                                
                                # 1. Save Heatmap Image
                                if 'heatmap_image' in payload:
                                    try:
                                        heatmap_data = base64.b64decode(payload['heatmap_image'])
                                        with open("final_heatmap.png", "wb") as f:
                                            f.write(heatmap_data)
                                        print("✅ Heatmap saved to final_heatmap.png")
                                    except Exception as e:
                                        print(f"Error saving heatmap: {e}")

                                # 2. Save CSV Summary
                                if 'session_csv' in payload:
                                    try:
                                        with open("final_summary.csv", "w") as f:
                                            f.write(payload['session_csv'])
                                        print("✅ Session summary saved to final_summary.csv")
                                    except Exception as e:
                                        print(f"Error saving CSV: {e}")
                                
                                # Signal the application to shut down
                                running = False
                                break # Exit the receiving loop
                            else:
                                # For other messages like tracking updates, just print them
                                print(f"Received: {message}")

                        except asyncio.TimeoutError:
                            continue
                        except websockets.exceptions.ConnectionClosed:
                            print("Network thread: Connection closed by server.")
                            break

                # Run both tasks concurrently
                await asyncio.gather(send_frames(), receive_messages())
        except Exception as e:
            print(f"Network thread error: {e}")
        finally:
            running = False
            print("Network thread finished.")

    asyncio.run(main_network_loop())


if __name__ == "__main__":
    """
    The Main Thread. It handles all GUI operations (OpenCV).
    """
    frame_queue = queue.Queue(maxsize=10)
    network_thread = threading.Thread(target=network_thread_entrypoint, args=(frame_queue,), daemon=True)
    network_thread.start()

    print("Starting webcam... Press 'q' in the window to stop.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        running = False
    else:
        while running:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Webcam Feed (Press 'q' to stop)", frame)

            if not frame_queue.full():
                frame_queue.put(frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                print("'q' pressed. Shutting down...")
                running = False
                break
    
    print("Waiting for network thread to finish...")
    network_thread.join(timeout=3.0)

    cap.release()
    cv2.destroyAllWindows()
    print("Application finished.")