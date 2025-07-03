import cv2
import requests
import base64
import numpy as np
import time
import os

# --- Configuration ---
API_URL = "http://127.0.0.1:8000"
SAVE_DIR = "client_output" # Directory to save files received from the server

def frame_to_base64(frame: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')
def base64_to_frame(b64_string: str) -> np.ndarray:
    img_bytes = base64.b64decode(b64_string)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return frame

def save_exported_files(session_id: str):
    print("⬇️  Requesting exported files...")
    try:
        response = requests.post(f"{API_URL}/session/{session_id}/export", timeout=15) # Add a timeout
        response.raise_for_status()
        export_data = response.json()
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        files_received = export_data.get("files", [])
        if not files_received:
            print("⚠️  Server did not return any files to save.")
            return

        for file_info in files_received:
            filename = file_info.get("filename")
            content_b64 = file_info.get("content_b64")
            save_path = os.path.join(SAVE_DIR, filename)
            try:
                content_bytes = base64.b64decode(content_b64)
                with open(save_path, "wb") as f:
                    f.write(content_bytes)
                print(f"✅ File saved locally: {save_path}")
            except Exception as e:
                print(f"❌ Error saving file {filename}: {e}")
                
    except requests.exceptions.Timeout:
        print("❌ Error: The request to the export endpoint timed out.")
    except requests.exceptions.RequestException as e:
        # Check if the response has JSON before trying to print it
        try:
            error_detail = e.response.json()
        except:
            error_detail = e.response.text
        print(f"❌ Error exporting files: {error_detail}")

def main():
    print("🚀 Starting new session...")
    try:
        response = requests.post(f"{API_URL}/session/start", json={"session_name": "test_client_session"})
        response.raise_for_status()
        session_id = response.json().get("session_id");
        if not session_id: print("❌ Error: Could not get session_id."); return
        print(f"✅ Session started. ID: {session_id}")
    except requests.exceptions.RequestException as e:
        print(f"❌ API connection error: {e}. Is the server running?"); return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("❌ Error: Could not open webcam."); return
    cv2.namedWindow("API Test Client", cv2.WINDOW_NORMAL)
    
    tracking = False
    
    print("\n--- Controls ---\n  't' - Start Tracking\n  's' - Stop Tracking, Analyze & Export\n  'q' - Quit\n----------------\n")

    while True:
        ret, frame = cap.read();
        if not ret: break
        display_frame = frame.copy()
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'): break
        elif key == ord('t') and not tracking:
            print("\n▶️  Sending START TRACKING request...")
            try:
                requests.post(f"{API_URL}/session/{session_id}/start_tracking").raise_for_status()
                print("✅ Tracking started on server."); tracking = True
            except requests.exceptions.RequestException as e: print(f"❌ Error: {e.response.json()}")
        elif key == ord('s') and tracking:
            print("\n⏹️  Sending STOP TRACKING request...")
            try:
                response = requests.post(f"{API_URL}/session/{session_id}/stop_tracking", json={"frame": frame_to_base64(frame)})
                response.raise_for_status()
                tracking = False
                print("✅ Swing analysis complete!"); print("--- Final Metrics ---"); [print(f"  {k}: {v}") for k, v in response.json().get("data", {}).items()]
                save_exported_files(session_id)
            except requests.exceptions.RequestException as e:
                print(f"❌ Error stopping tracking: {e.response.json()}"); tracking = False

        if tracking:
            try:
                start_time = time.time()
                response = requests.post(f"{API_URL}/session/{session_id}/process_frame", json={"frame": frame_to_base64(frame)})
                response.raise_for_status()
                latency = (time.time() - start_time) * 1000
                display_frame = base64_to_frame(response.json().get("processed_frame"))
                cv2.putText(display_frame, f"API Latency: {latency:.0f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except requests.exceptions.RequestException as e:
                print(f"❌ Error processing frame: {e.response.json()}"); tracking = False

        cv2.imshow("API Test Client", display_frame)

    print("🔌 Cleaning up..."); cap.release(); cv2.destroyAllWindows()
    print("👋 Client shut down.")

if __name__ == "__main__":
    main()