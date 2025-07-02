import asyncio
import websockets
import cv2
import json
import base64

SERVER_URL = "ws://localhost:8000/ws/v1/swing_analysis"

async def run_test():
    """
    Connects to the server, streams webcam frames, and handles responses.
    """
    async with websockets.connect(SERVER_URL) as websocket:
        print("Connected to Swingman API server.")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        # Coroutine to listen for messages from the server
        async def receive_messages():
            try:
                while True:
                    message_str = await websocket.recv()
                    message = json.loads(message_str)
                    msg_type = message.get("type")

                    if msg_type == "tracking_update":
                        efficiency = message['payload']['metrics'].get('efficiency_score', 0)
                        power = message['payload']['metrics'].get('power_score', 0)
                        print(f"LIVE DATA | Efficiency: {efficiency}%, Power: {power}%")
                    
                    elif msg_type == "session_end":
                        print("\n--- FINAL ARTIFACTS RECEIVED ---")
                        # 1. Save Heatmap
                        heatmap_b64 = message['payload']['heatmap_image']
                        heatmap_data = base64.b64decode(heatmap_b64)
                        with open("final_heatmap.png", "wb") as f:
                            f.write(heatmap_data)
                        print("✅ Heatmap saved to final_heatmap.png")
                        
                        # 2. Save CSV
                        csv_data = message['payload']['session_csv']
                        with open("final_summary.csv", "w") as f:
                            f.write(csv_data)
                        print("✅ Session summary saved to final_summary.csv")
                        
                        break # End the listener loop
                        
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by server.")

        # Coroutine to send frames to the server
        async def send_frames():
            print("\nStreaming frames... Press 'q' in the webcam window to stop.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Display the frame locally so you can see what's being sent
                cv2.imshow("Webcam Feed (Press 'q' to stop)", frame)

                # Encode frame as JPEG for efficient network transfer
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                
                await websocket.send(buffer.tobytes())

                # Poll for 'q' key to quit, and give server time to process
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                await asyncio.sleep(1/30) # Stream at ~30 FPS

            # Signal stop to the server
            print("Stopping session and requesting final artifacts...")
            await websocket.send(json.dumps({"action": "stop"}))
            
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()

        # Run both tasks concurrently
        await asyncio.gather(
            send_frames(),
            receive_messages()
        )
        print("\nTest client finished.")


if __name__ == "__main__":
    asyncio.run(run_test())