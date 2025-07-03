import asyncio
import websockets
import cv2
import json
import base64
import threading

# --- Configuration ---
SERVER_URL = "ws://localhost:8000/ws/v1/swing_analysis"
JPEG_QUALITY = 80 # 0-100, higher is better quality, larger size

# A thread-safe flag to signal when to stop
running = True

def capture_and_display_thread(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """
    This function runs in a separate thread and handles all blocking
    OpenCV operations.
    """
    global running
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        running = False
        return

    print("\nWebcam feed running. Press 'q' in the window to stop.")

    while running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Display the frame locally. This window will be responsive.
        cv2.imshow("Webcam Feed (Press 'q' to stop)", frame)

        # Put the frame into the asyncio queue in a thread-safe manner
        if not queue.full():
            loop.call_soon_threadsafe(queue.put_nowait, frame)
        
        # Check for 'q' key to quit. This is now highly responsive.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' pressed, signaling shutdown...")
            running = False
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam thread finished.")


async def network_tasks(queue: asyncio.Queue):
    """
    This function runs in the main asyncio thread and handles all
    network communication.
    """
    global running
    async with websockets.connect(SERVER_URL) as websocket:
        print(f"✅ Connected to server at {SERVER_URL}")

        # Task to send frames from the queue to the server
        async def send_frames():
            while running:
                try:
                    # Get a frame from the queue, waiting up to 0.1s
                    frame = await asyncio.wait_for(queue.get(), timeout=0.1)
                    
                    # Encode and send
                    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                    await websocket.send(buffer.tobytes())
                    
                    queue.task_done()
                except asyncio.TimeoutError:
                    # If queue is empty, just continue. This keeps the loop alive.
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break
            
            # After the loop, signal the server to stop the session
            if not websocket.closed:
                print("Sending stop_session command to server...")
                # Use a different action from your final API spec if needed
                await websocket.send(json.dumps({"action": "stop_session"}))


        # Task to receive messages from the server
        async def receive_messages():
            while running:
                try:
                    message_str = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    message = json.loads(message_str)
                    
                    # Process different message types
                    if message.get("type") == "session_end":
                        print("\n--- FINAL ARTIFACTS RECEIVED ---")
                        payload = message['payload']
                        with open("final_heatmap.png", "wb") as f:
                            f.write(base64.b64decode(payload['heatmap_image']))
                        print("✅ Heatmap saved to final_heatmap.png")
                        with open("final_summary.csv", "w") as f:
                            f.write(payload['session_csv'])
                        print("✅ Session summary saved to final_summary.csv")
                        break # Exit after receiving final artifacts
                    else:
                        # Print any other messages like tracking updates
                        print(f"Received: {message}")

                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed by server.")
                    break

        # Run both network tasks concurrently
        await asyncio.gather(send_frames(), receive_messages())


async def main():
    """ Main entry point for the application. """
    # A queue to pass frames from the GUI thread to the network thread
    frame_queue = asyncio.Queue(maxsize=10)
    
    # Get the current asyncio event loop
    loop = asyncio.get_running_loop()

    # Start the blocking GUI function in a separate thread
    # This is the key to a responsive UI
    gui_thread = threading.Thread(target=capture_and_display_thread, args=(frame_queue, loop), daemon=True)
    gui_thread.start()

    # Run the non-blocking network tasks
    await network_tasks(frame_queue)

    print("\nTest client finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program interrupted.")