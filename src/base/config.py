import os
from dotenv import load_dotenv

load_dotenv()

object_detection_config = {
    "exercise_name": None,
    "exercise_config": [],
    "device": os.getenv('DEVICE'),
    "device_ids": os.getenv('DEVICE_IDS'),
    "model": "./models/yolo/yolo11n.pt",
    "confidence_threshold": 0.1,
    "iou_threshold": 0.6,
    "tracker_activation_threshold": 0.45,
    "tracker_matching_threshold": 0.95,
    "tracker_consecutive_frames": 3
}