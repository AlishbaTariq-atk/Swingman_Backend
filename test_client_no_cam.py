import asyncio
import websockets
import json
import cv2
import base64
import threading
import queue
import time

# --- Configuration ---
SERVER_URL = "ws://localhost:8000/ws/v1/swing_analysis" # Ensure port matches server
TEST_IMAGE_PATH = "test_swing.jpg"
JPEG_QUALITY = 80

def network_thread_entrypoint(frame_queue: queue.Queue):
    """
    The entry point for our background network thread. Runs the asyncio event loop.
    This function is IDENTICAL to the one in the interactive webcam client.
    """
    async def main_network_loop():
        try:
            async with websockets.connect(SERVER_URL) as websocket:
                print(f"✅ Network thread connected to {SERVER_URL}")

                while True:
                    try:
                        # Block until the main thread puts a frame or a command in the queue
                        item = frame_queue.get(timeout=1.0)
                        
                        # Check for the shutdown sentinel
                        if item is None:
                            print("Network thread received shutdown signal.")
                            break
                        
                        # Check for JSON commands (as text)
                        if isinstance(item, str):
                            await websocket.send(item)
                            continue
                        
                        # Otherwise, it's an image frame (as bytes)
                        await websocket.send(item)
                        
                        # Try to receive any incoming messages without blocking for too long
                        try:
                            message_str = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                            message = json.loads(message_str)
                            print(f"Received: {message.get('type')}")
                        except asyncio.TimeoutError:
                            pass

                    except queue.Empty:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        print("Network thread: Connection closed prematurely.")
                        return

                # --- Shutdown Handshake ---
                if not websocket.closed:
                    print("Sending 'stop_session' command...")
                    await websocket.send(json.dumps({"action": "stop_session"}))
                    
                    print("Waiting for final artifacts...")
                    try:
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
                        print("Error: Timed out waiting for final artifacts.")

        except Exception as e:
            print(f"Network thread error: {e}")
        finally:
            print("Network thread finished.")

    asyncio.run(main_network_loop())


if __name__ == "__main__":
    """
    The Main Thread. Simulates user actions (starting/stopping swings) and
    produces image frames for the network thread.
    """
    frame_queue = queue.Queue(maxsize=30)
    network_thread = threading.Thread(target=network_thread_entrypoint, args=(frame_queue,), daemon=True)
    network_thread.start()

    time.sleep(2.0)

    print(f"Loading test image from {TEST_IMAGE_PATH}...")
    try:
        frame = cv2.imread(TEST_IMAGE_PATH)
        assert frame is not None
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        image_bytes = buffer.tobytes()
    except:
        print(f"Error: Could not load or encode test image at '{TEST_IMAGE_PATH}'.")
        # Signal shutdown if image fails to load
        frame_queue.put(None)
    else:
        # --- Main Simulation Loop ---
        try:
            # Simulate 2 full swings
            for i in range(1, 3):
                print(f"\n--- SIMULATING SWING {i} ---")
                
                # Send start command
                frame_queue.put(json.dumps({"action": "start_swing"}))
                print("Queued: start_swing")
                
                # Send 30 frames for the swing
                for _ in range(30):
                    if not frame_queue.full():
                        frame_queue.put(image_bytes)
                    time.sleep(1/30) # Simulate ~30 FPS
                
                # Send stop command
                frame_queue.put(json.dumps({"action": "stop_swing"}))
                print("Queued: stop_swing")
                time.sleep(0.5) # Give time for analysis to come back

            # Signal the network thread to shut down
            print("\nSignaling shutdown...")
            frame_queue.put(None)

        except KeyboardInterrupt:
            print("\nInterrupted by user. Signaling shutdown...")
            frame_queue.put(None)

    # Wait for the network thread to finish its clean shutdown
    print("Waiting for network thread to clean up...")
    network_thread.join(timeout=6.0)
    
    print("Application finished.")