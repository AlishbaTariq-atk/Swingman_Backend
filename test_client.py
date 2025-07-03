import asyncio
import websockets
import cv2
import json
import base64
import threading
import queue

SERVER_URL = "ws://localhost:8000/ws/v1/swing_analysis" # Ensure port is correct
JPEG_QUALITY = 80

# Use a threading.Event for safe cross-thread communication
shutdown_event = threading.Event()

def network_thread_entrypoint(frame_queue: queue.Queue):
    """ The entry point for our background network thread. """
    async def main_network_loop():
        try:
            async with websockets.connect(SERVER_URL) as websocket:
                print(f"✅ Network thread connected.")

                # --- Main processing loop ---
                # This loop handles both sending and receiving concurrently
                while not shutdown_event.is_set():
                    # 1. Try to send a frame if one is available
                    try:
                        frame = frame_queue.get_nowait()
                        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                        await websocket.send(buffer.tobytes())
                    except queue.Empty:
                        # No frame to send, this is normal
                        pass
                    except websockets.exceptions.ConnectionClosed:
                        break

                    # 2. Try to receive a message
                    try:
                        message_str = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                        message = json.loads(message_str)
                        # We only care about printing the type for live feedback
                        print(f"Received: {message.get('type')}")
                    except asyncio.TimeoutError:
                        # No message received, this is normal
                        pass
                    except websockets.exceptions.ConnectionClosed:
                        break

                # --- Shutdown handshake ---
                # The loop has ended, now we perform a clean shutdown.
                if not websocket.closed:
                    # 1. Tell the server we are done
                    print("Sending 'stop_session' command...")
                    await websocket.send(json.dumps({"action": "stop_session"}))
                    
                    # 2. Wait for the final artifacts message
                    print("Waiting for final artifacts...")
                    try:
                        # Wait up to 5 seconds for the server to process and respond
                        response_str = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        message = json.loads(response_str)
                        if message.get("type") == "session_end":
                            print("\n--- FINAL ARTIFACTS RECEIVED ---")
                            payload = message.get("payload", {})
                            with open("final_heatmap.png", "wb") as f: f.write(base64.b64decode(payload['heatmap_image']))
                            print("✅ Heatmap saved.")
                            with open("final_summary.csv", "w") as f: f.write(payload['session_csv'])
                            print("✅ CSV summary saved.")
                    except asyncio.TimeoutError:
                        print("Error: Timed out waiting for final artifacts from server.")
                    except websockets.exceptions.ConnectionClosed:
                        print("Connection closed while waiting for final artifacts.")

        except Exception as e:
            print(f"Network thread error: {e}")
        finally:
            print("Network thread finished.")

    asyncio.run(main_network_loop())

if __name__ == "__main__":
    frame_queue = queue.Queue(maxsize=30)
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
                shutdown_event.set()
    
    print("Waiting for network thread to clean up...")
    network_thread.join(timeout=6.0) # Wait for network thread to finish its work
    
    cap.release()
    cv2.destroyAllWindows()
    print("Application finished.")