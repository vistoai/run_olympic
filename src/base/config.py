import os
from dotenv import load_dotenv

load_dotenv()

person_detection_set_config = {
    "exercise_name": None,
    "exercise_config": [],
}

person_detection_model_config = {
    "model": "./models/yolo/yolo11n.pt",
    "task": "detect",
}

person_detection_inference_config = {
    "conf": 0.1,
    "iou": 0.6,
    "classes": [0],
    "device": [0],
    "half": False,
    "agnostic_nms": True,
    "stream": False,
}

person_detection_tracker_config = {
    "track_activation_threshold": 0.45,
    "minimum_matching_threshold": 0.95,
    "minimum_consecutive_frames": 3,
}

paddleocr_detection_recognition_model_config = {
    "use_angle_cls": True,
    "lang": "en",
    "use_gpu": True,
    "gpu_id": 0,
    "det_model_dir": f"{os.getenv('HOME')}/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer",
    "rec_model_dir": f"{os.getenv('HOME')}/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer",
    "cls_model_dir": f"{os.getenv('HOME')}/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer",
    "show_log": False
}