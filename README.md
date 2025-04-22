# Vehicle Tracking with YOLOv8 and Centroid Tracker

## Problem Statement

This project implements real-time vehicle tracking using YOLOv8 for vehicle detection and a Centroid Tracker to track vehicles across video frames.

The goal is to build a system for counting and tracking vehicles, useful in traffic monitoring, smart cities, and vehicle flow analysis.

## Project Structure

The repository is organized as follows:

```
vehicle-tracking-yolov8-flask/
├── app.py                        # Flask web app for uploading and processing videos
├── index.html                    # Frontend for video upload and viewing results
├── uploads/                      # Temporary storage for uploaded videos
├── yolov8n.pt                    # Pre-trained YOLOv8 model
├── requirements.txt              # Dependencies
└── vehicle_tracking_yolov8.ipynb # Jupyter Notebook for testing and implementation
```

## Explanation

### YOLOv8 (You Only Look Once v8)

- Real-time object detection model used to detect vehicles in video frames.
- Pre-trained and classifies vehicles into:
  - Car (2)
  - Motorcycle (3)
  - Bus (5)
  - Truck (7)

### Centroid Tracker

- Tracks detected vehicles using the centroid (center point) of bounding boxes.
- Assigns unique IDs to each vehicle across frames based on centroid proximity.

## Methods Used

1. **YOLOv8 Detection**  
   - Detects vehicles and outputs bounding boxes and class labels.
   - Classes considered: car, motorcycle, bus, truck.

2. **Centroid Tracking**  
   - Uses bounding box midpoints to track vehicles.
   - Assigns unique IDs across frames.

3. **Flask Web App**  
   - Allows video uploads and processes them in real-time.
   - Returns:
     - Total unique vehicles detected.
     - Vehicle count per frame.

## Jupyter Notebook

The `vehicle_tracking_yolov8.ipynb` notebook covers:

- Loading and testing YOLOv8 on sample video.
- Frame-wise vehicle detection.
- Implementing Centroid Tracker.
- Vehicle counting and result visualization.

## Web App Flow

1. User uploads a video through the web interface.
2. YOLOv8 detects vehicles, Centroid Tracker tracks them.
3. App returns:
   - Total unique vehicles
   - Per-frame vehicle counts
4. Frontend displays results dynamically.

## Dataset

- Sample video in MP4 format with traffic scenes.
- Any traffic video can be used for testing.

## Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/archanadby05/traffic_vehicle_counter.git
   cd traffic_vehicle_counter
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask app:
   ```bash
   python app.py
   ```

4. Open in browser:
   ```
   http://127.0.0.1:5000
   ```

5. Upload a video and view results.
