import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import *
from util import read_license_plate
from datetime import datetime, timezone

from pymongo import MongoClient

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = ""

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Connect to MongoDB
# client = MongoClient('mongodb://localhost:27017/')
db = client['license_plate_database']
collection = db['license_plate_collection']

mot_tracker = Sort()
highest_scores = {}

# Load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('models/license_plate_detector.pt')

# Open the camera
cap = cv2.VideoCapture(1)  # Use 0 for the default camera

# Initialize variables
frame_nmr = 0
vehicles = [2, 3, 7, 5]
skip_frames = 10  # Adjust this value as needed, it determines how many frames to skip
ret = True
while ret and frame_nmr < 1200:
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        break
    cv2.imshow('Webcam', frame)


    # Process frames at regular intervals
    if frame_nmr % skip_frames == 0:
        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Track vehicles
        #track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license to car
            # xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            car_id=2
            # if car_id != -1:
                # Crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # Process license plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255,
                                                              cv2.THRESH_BINARY_INV)

                # Read license plate number
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                # Store results
            if license_plate_text is not None:
                    # Update highest scoring license plate text for each car
                    if car_id not in highest_scores or score > highest_scores[car_id]['score']:
                        highest_scores[car_id] = {'text': license_plate_text, 'score': score}

                    # Record camera number, location, and timestamp for the current frame
                    camera_number = "Camera 3"  
                    location = "28°36'52.8N 77°13'41.0E"  # replace with actual location
                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    timestamp = datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')

                    # Store these values in the results dictionary
                    highest_scores[car_id]['camera_number'] = camera_number
                    highest_scores[car_id]['location'] = location
                    highest_scores[car_id]['timestamp'] = timestamp

    # Store results to MongoDB every 30 frames (adjust as needed)
    if frame_nmr % (10*3) == 0:
        # Store results in MongoDB
        for car_id, data in highest_scores.items():
            document = {
                'frame_nmr': frame_nmr,
                'car_id': car_id,
                'license_plate_text': highest_scores[car_id].get('text', 'N/A'),
                'license_number_score': highest_scores[car_id].get('score', 'N/A'),
                'camera_number': highest_scores[car_id].get('camera_number', 'N/A'),
                'location': highest_scores[car_id].get('location', 'N/A'),
                'timestamp': highest_scores[car_id].get('timestamp', 'N/A')
            }
            collection.insert_one(document)

        # Clear highest scores for next batch
        highest_scores = {}

    frame_nmr += 1
        # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# Close the MongoDB connection
client.close()
