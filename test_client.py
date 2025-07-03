import cv2
import requests
import base64
import numpy as np
import time

# --- Configuration ---
API_URL = "http://127.0.0.1:8000"  # URL of your running FastAPI server

def frame_to_base64(frame: np.ndarray) -> str:
    """Encodes an OpenCV image (frame) to a Base64 string."""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def base64_to_frame(b64_string: str) -> np.ndarray:
    """Decodes a Base64 string to an OpenCV image (frame)."""
    img_bytes = base64.b64decode(b64_string)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return frame

def main():
    """Main function to run the test client."""
    
    # 1. Start a new session
    print("🚀 Starting new session...")
    try:
        response = requests.post(f"{API_URL}/session/start", json={"session_name": "test_client_session"})
        response.raise_for_status()  # Raises an exception for bad status codes (4xx or 5xx)
        session_data = response.json()
        session_id = session_data.get("session_id")
        if not session_id:
            print("❌ Error: Could not get session_id from server.")
            return
        print(f"✅ Session started successfully. Session ID: {session_id}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to the API: {e}")
        print("   Please ensure the `uvicorn api_server:app --reload` server is running.")
        return

    # 2. Setup camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    cv2.namedWindow("API Test Client", cv2.WINDOW_NORMAL)
    
    tracking = False
    
    print("\n--- Controls ---")
    print("  't' - Start Tracking")
    print("  's' - Stop Tracking & Analyze")
    print("  'q' - Quit")
    print("----------------\n")


    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error reading frame from webcam.")
            break

        display_frame = frame.copy()
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        
        elif key == ord('t') and not tracking:
            # 3. Start tracking
            print("\n▶️  Sending START TRACKING request...")
            try:
                response = requests.post(f"{API_URL}/session/{session_id}/start_tracking")
                response.raise_for_status()
                print("✅ Tracking started on server.")
                tracking = True
            except requests.exceptions.RequestException as e:
                print(f"❌ Error starting tracking: {e.response.json()}")

        elif key == ord('s') and tracking:
            # 5. Stop tracking
            print("\n⏹️  Sending STOP TRACKING request...")
            try:
                b64_frame = frame_to_base64(frame)
                response = requests.post(
                    f"{API_URL}/session/{session_id}/stop_tracking",
                    json={"frame": b64_frame}
                )
                response.raise_for_status()
                tracking = False
                final_data = response.json()
                print("✅ Swing analysis complete!")
                print("--- Final Metrics ---")
                for k, v in final_data.get("data", {}).items():
                    print(f"  {k}: {v}")
                print("---------------------\n")
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Error stopping tracking: {e.response.json()}")
                tracking = False # Reset state

        if tracking:
            # 4. Process frame
            try:
                b64_frame = frame_to_base64(frame)
                
                start_time = time.time()
                response = requests.post(
                    f"{API_URL}/session/{session_id}/process_frame",
                    json={"frame": b64_frame}
                )
                response.raise_for_status()
                end_time = time.time()

                data = response.json()
                processed_b64 = data.get("processed_frame")
                
                # Update the display frame with the processed one from the server
                display_frame = base64_to_frame(processed_b64)
                
                # Display network latency
                latency = (end_time - start_time) * 1000
                cv2.putText(display_frame, f"API Latency: {latency:.0f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Error processing frame: {e.response.json()}")
                # Stop tracking if a processing error occurs
                tracking = False

        cv2.imshow("API Test Client", display_frame)

    print("🔌 Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Client shut down.")

if __name__ == "__main__":
    main()