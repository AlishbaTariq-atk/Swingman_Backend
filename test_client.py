import asyncio
import websockets
import cv2
import json
import threading
import queue  # Use the standard thread-safe queue

# --- Configuration ---
SERVER_URL = "ws://localhost:8000/ws/v1/swing_analysis"
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
                            # Get a frame from the queue without blocking the event loop
                            frame = frame_queue.get_nowait()
                            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                            await websocket.send(buffer.tobytes())
                            frame_queue.task_done()
                        except queue.Empty:
                            # Queue is empty, wait briefly to yield control
                            await asyncio.sleep(0.01)
                        except websockets.exceptions.ConnectionClosed:
                            break
                    
                    if not websocket.closed:
                        print("Network thread: Sending stop_session command...")
                        await websocket.send(json.dumps({"action": "stop_session"}))

                # Task to continuously receive messages from the server
                async def receive_messages():
                    while running:
                        try:
                            message_str = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                            # In a real app, you would process this. For now, we just print.
                            print(f"Received: {json.loads(message_str)}")
                        except asyncio.TimeoutError:
                            continue # No message received, continue loop
                        except websockets.exceptions.ConnectionClosed:
                            print("Network thread: Connection closed by server.")
                            break

                # Run both tasks concurrently
                await asyncio.gather(send_frames(), receive_messages())
        except Exception as e:
            print(f"Network thread error: {e}")
        finally:
            # Ensure the main loop knows we are done
            running = False
            print("Network thread finished.")

    # Start the asyncio event loop for this background thread
    asyncio.run(main_network_loop())


if __name__ == "__main__":
    """
    The Main Thread. It handles all GUI operations (OpenCV).
    """
    # A standard, thread-safe queue to pass frames to the network thread
    frame_queue = queue.Queue(maxsize=10)

    # Create and start the background thread for networking
    network_thread = threading.Thread(target=network_thread_entrypoint, args=(frame_queue,), daemon=True)
    network_thread.start()

    # --- OpenCV Main Loop ---
    print("Starting webcam... Press 'q' in the window to stop.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        running = False
    else:
        while running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Display the webcam feed. This is now on the main thread.
            cv2.imshow("Webcam Feed", frame)

            # Put the frame into the queue for the network thread to send
            if not frame_queue.full():
                frame_queue.put(frame)

            # Check for the 'q' key to quit
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                print("'q' pressed. Shutting down...")
                running = False
                break
    
    # --- Cleanup ---
    print("Waiting for network thread to finish...")
    network_thread.join(timeout=2.0) # Wait for the thread to exit gracefully

    cap.release()
    cv2.destroyAllWindows()
    print("Application finished.")