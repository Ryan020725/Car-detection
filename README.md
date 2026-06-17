# Video sensing Car-detection

## Overview

1. Data from https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4

2. YOLOv10 :

It is NMS-Free, which eliminates the post-processing "latency tax" typically found in models like YOLOv8. Which is critical for maintaining high performance in CPU-bound environments like GitHub Codespaces


3. The overall structure of the code follows the Producer-Consumer architecture mandated to handle high-speed video inputs without system crashes.

Initializes a queue.Queue(maxsize=1), which is the specific mechanism required to implement the "Drop-and-Replace" logic. This ensures the system only ever processes the "freshest" data, preventing the "latency drift" and Out-of-Memory (OOM) errors 

## Learning objectives

1. The configuration explicitly sets `MODEL_PATH = "yolov10n.pt"`.

2. Defines separate frame_producer and frame_consumer functions and manages them using the threading module.

3. The code includes a `get_mock_sensor_data() ` function to provide high-frequency scalar data to be joined with the vision results.

## Instructions
1. **Setup**:
   `pip install opencv-python `
   `pip install ultralytics numpy opencv-python `


  

## Expected output

- The continuous lines showing Processed: `X vehicles | Latency: XX.XXms` represent the output of the YOLOv10 NMS-Free model
- Producer-Consumer Architecture: The final lines of the log `([Consumer] Stopped cleanly, [Producer] Video playback stopped)`confirm that separate threads were managing the data flow independently
- The bottom section of the image shows the result of a shutdown signal (likely a `KeyboardInterrupt` or `Ctrl+C`

