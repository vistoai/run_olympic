import os
import sys
import json
import time
import random
from functools import partial
from multiprocessing import Queue, Process
from typing import Any, List, Optional, Union
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), './src')))

import cv2
import numpy as np
import supervision as sv
from dotenv import load_dotenv
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame

from base.config import (
    person_detection_set_config,
    person_detection_model_config,
    person_detection_inference_config,
    person_detection_tracker_config, 
    paddleocr_detection_recognition_model_config
)
from base.utils import filter_paddleocr_coordinate_to_person_detection_coordinate
from app.running_olympic_roboflow_app import RunningOlympicRoboflowApp

load_dotenv()

# Annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def on_prediction(
    predictions: Union[Any, List[Optional[dict]]], 
    video_frame: Union[VideoFrame, List[Optional[VideoFrame]]],
) -> None:
    if not issubclass(type(predictions), list):
      predictions = [predictions]
      video_frame = [video_frame]
    
    print(f"Predictions: {len(predictions)}")
    print(f"Video Frame: {len(video_frame)}")
    for prediction, v in zip(predictions, video_frame):
        frame = v.image
        if prediction is None:
            continue
        
        # Get the prediction
        person_detection_prediction, ocr_prediction = prediction

        print("Person Detection Prediction")
        print(person_detection_prediction)
        print("Person Detection Prediction XYXY")
        print(person_detection_prediction.xyxy)
        print("OCR Prediction")
        print(ocr_prediction)

        filtered_ocr_prediction = filter_paddleocr_coordinate_to_person_detection_coordinate(
            person_detection=person_detection_prediction,
            paddleocr_detection=ocr_prediction
        )
        ocr_prediction = ocr_prediction[np.where(filtered_ocr_prediction == True)]

        print("OCR Prediction After Filtering")
        print(ocr_prediction)

        # Create annotation
        labels = [f"#{class_id}" for class_id in person_detection_prediction.class_id]
        annotated_frame = box_annotator.annotate(
            scene=frame.copy(),
            detections=person_detection_prediction,
        )
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame.copy(),
            detections=ocr_prediction,
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame.copy(),
            detections=person_detection_prediction,
            labels=labels,
        )

        cv2.imwrite(f"media/running_result_image/annotated_frame_{random.randint(1, 1000000)}.jpg", annotated_frame)
        

        


    
if __name__ == "__main__":
    running_olympic_roboflow_app = RunningOlympicRoboflowApp(
        person_detection_set_config=person_detection_set_config,
        person_detection_model_config=person_detection_model_config,
        person_detection_inference_config=person_detection_inference_config,
        person_detection_tracker_config=person_detection_tracker_config,
        paddleocr_detection_recognition_model_config=paddleocr_detection_recognition_model_config
    )

    pipeline = InferencePipeline.init_with_custom_logic(
        video_reference=[
            "./media/video/running.mp4", 
            # "./media/video/running.mp4",
            # "./media/video/running.mp4",
            # "./media/video/running.mp4",
            # "./media/video/running.mp4",
            # "./media/video/running.mp4",
            # "./media/video/running.mp4",
            # "./media/video/running.mp4",
            # "./media/video/running.mp4",
            # "./media/video/running.mp4",
            # "./media/video/running.mp4",
            # "./media/video/running.mp4",
            # "./media/video/running.mp4",
            # "./media/video/running.mp4",
            # "./media/video/running.mp4",
            # "./media/video/running.mp4",
        ],
        on_video_frame=running_olympic_roboflow_app.infer,
        on_prediction=on_prediction,
        max_fps=10
    )

    # Start the pipeline
    pipeline.start()
    pipeline.join()