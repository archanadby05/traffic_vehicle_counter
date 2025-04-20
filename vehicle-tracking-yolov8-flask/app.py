from flask import Flask, render_template, request, jsonify
import cv2
from ultralytics import YOLO
import os
from collections import defaultdict
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Vehicle classes
vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# CentroidTracker class
class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_video', methods=['POST'])
def process_video():
    # Handle file upload
    video_file = request.files['video']
    video_path = os.path.join("uploads", video_file.filename)
    video_file.save(video_path)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_unique_ids = set()
    class_id_map = {}
    per_frame_stats = defaultdict(int)
    
    # Initialize CentroidTracker
    ct = CentroidTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame)[0]

        rects = []
        boxes_info = []

        for box in results.boxes:
            cls_id = int(box.cls)
            if cls_id in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                rects.append((x1, y1, x2, y2))
                boxes_info.append(((x1, y1, x2, y2), cls_id))

        # Update vehicle tracking with the centroid tracker
        objects = ct.update(rects)

        for object_id, centroid in objects.items():
            if object_id < len(boxes_info):
                (x1, y1, x2, y2), cls_id = boxes_info[object_id]
                class_name = vehicle_classes[cls_id]
                per_frame_stats[class_name] += 1
                total_unique_ids.add(object_id)
                class_id_map[object_id] = class_name

    cap.release()

    # Return statistics as JSON
    return jsonify({
        "unique_vehicles": len(total_unique_ids),
        "per_frame_stats": dict(per_frame_stats)
    })


if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)