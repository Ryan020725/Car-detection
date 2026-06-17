import os
import urllib.request
import cv2
import time
import queue
import threading
import json
from ultralytics import YOLO
import paho.mqtt.client as mqtt

MODEL_PATH = "yolov10n.pt"  
MQTT_BROKER = "localhost"
LOCAL_LOG = "local.jsonl"
frame_queue = queue.Queue(maxsize=1)  

stop_event = threading.Event()
mqtt_connected = False

# =================================================================
# INTEL IOT VIDEO DOWNLOADER (UPDATED FOR VEHICLES)
# =================================================================
def get_intel_car_video():
    """Downloads the Intel IoT car detection sample video if missing."""
    url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"
    video_filename = "intel_cars.mp4"
    
    if not os.path.exists(video_filename):
        print(f"[*] Downloading Intel IoT Car video from GitHub...")
        try:
            urllib.request.urlretrieve(url, video_filename)
            print(f"[+] Download complete! Saved as: {video_filename}")
        except Exception as e:
            print(f"[-] Failed to download video: {e}")
            raise e
    else:
        print(f"[+] Found existing sample video: {video_filename}")
        
    return video_filename

def get_mock_sensor_data():
    return {"water_level": 5.2, "timestamp": time.time()}

def frame_producer(video_path):
    cap = cv2.VideoCapture(video_path)
    print(f"[Producer] Started video playback: {video_path}")
    
    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: 
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                frame_queue.get_nowait()  
                frame_queue.put_nowait(frame)  
            except queue.Empty:
                pass
                
        # Approximate 30 FPS pacing to match real video speed
        time.sleep(0.033)
        
    cap.release()
    print("[Producer] Video playback stopped.")

def frame_consumer():
    global mqtt_connected
    model = YOLO(MODEL_PATH)
    client = mqtt.Client()
    
    try:
        client.connect(MQTT_BROKER, 1883, 60)
        client.loop_start() 
        mqtt_connected = True
    except Exception:
        print("[Consumer] MQTT Broker offline, using local fallback.")

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue 
        
        start_time = time.time()
        results = model(frame, verbose=False)
        inference_time = (time.time() - start_time) * 1000 
        
        detections = len(results[0].boxes) if results else 0
        sensor_reading = get_mock_sensor_data()
        
        payload = {
            "timestamp": time.time(),
            "detections": detections,
            "sensor_data": sensor_reading,
            "latency_ms": inference_time
        }

        if mqtt_connected:
            try:
                client.publish("edge/vision_fusion", json.dumps(payload))
            except Exception:
                mqtt_connected = False 
        
        if not mqtt_connected:
            with open(LOCAL_LOG, "a") as f:
                f.write(json.dumps(payload) + "\n")
        
        print(f"Processed: {detections} vehicles | Latency: {inference_time:.2f}ms")

    if mqtt_connected:
        client.loop_stop()
        client.disconnect()
    print("[Consumer] Stopped cleanly.")


if __name__ == "__main__":
  
    video_input = get_intel_car_video()
    
   
    t1 = threading.Thread(target=frame_producer, args=(video_input,), daemon=True)
    t2 = threading.Thread(target=frame_consumer, daemon=True)
    
    t1.start()
    t2.start()
    
    try:
        print("[Main] Pipeline running with Intel Car Dataset. Press Ctrl+C to terminate...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[Main] Shutdown signal caught...")    
        stop_event.set() 
    
    t1.join()
    t2.join()
    print("[Main] Safe to exit.")
