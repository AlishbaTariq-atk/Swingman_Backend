import asyncio
import websockets
import json
import cv2
import base64
import os

# Configure via environment variable for flexibility
SERVER_URL = os.getenv("WEBSOCKET_URL", "ws://localhost:8000/ws/v1/swing_analysis")
TEST_IMAGE_PATH = "test_swing.jpg" 

async def run_swing_simulation():
    """ A clean test client that simulates a full user session with multiple swings. """
    try:
        frame = cv2.imread(TEST_IMAGE_PATH)
        assert frame is not None, f"Test image not found at {TEST_IMAGE_PATH}"
    except Exception as e:
        print(f"Error loading test image: {e}")
        return

    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()

    async with websockets.connect(SERVER_URL) as websocket:
        print(f"✅ Connected to server at {SERVER_URL}")

        for i in range(1, 3): # Simulate 2 swings
            print(f"\n--- SIMULATING SWING {i} ---")
            await websocket.send(json.dumps({"action": "start_swing"}))
            for _ in range(30):
                await websocket.send(image_bytes)
                await websocket.recv() # Receive and discard tracking updates
            
            await websocket.send(json.dumps({"action": "stop_swing"}))
            swing_result = json.loads(await websocket.recv())
            print(f"Received Swing {i} Analysis: {swing_result['payload']}")

        print("\n--- ENDING SESSION ---")
        await websocket.send(json.dumps({"action": "stop_session"}))
        final_artifacts = json.loads(await websocket.recv())
        
        if final_artifacts.get("type") == "session_end":
            print("\n--- FINAL ARTIFACTS RECEIVED ---")
            payload = final_artifacts['payload']
            with open("final_heatmap.png", "wb") as f:
                f.write(base64.b64decode(payload['heatmap_image']))
            print("✅ Heatmap saved to final_heatmap.png")

            with open("final_summary.csv", "w") as f:
                f.write(payload['session_csv'])
            print("✅ Session summary saved to final_summary.csv")

if __name__ == "__main__":
    asyncio.run(run_swing_simulation())