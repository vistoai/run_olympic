import os
import sys
import json
from typing import Any, List, Optional, Union
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), './src')))

from dotenv import load_dotenv
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame

from base.config import object_detection_config
from app.person_detection_roboflow_app import PersonDetectionRoboflowApp

load_dotenv()

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
        
        # print(f"Prediction: {prediction}")
        """
        Output:
        Prediction: Detections(xyxy=array([], shape=(0, 4), dtype=float32), mask=None, confidence=array([], dtype=float32), class_id=array([], dtype=int64), tracker_id=array([], dtype=int64), data={}, metadata={})
        """
        # print(f"Frame Shape: {frame.shape}")
        """
        Output:
        Frame Shape: (1080, 1920, 3)
        """
        

person_detection_app = PersonDetectionRoboflowApp(
    config=object_detection_config
)

pipeline = InferencePipeline.init_with_custom_logic(
    video_reference=[
        "./media/video/running.mp4", 
        "./media/video/running.mp4",
        "./media/video/running.mp4",
        "./media/video/running.mp4",
        "./media/video/running.mp4",
        "./media/video/running.mp4",
        "./media/video/running.mp4",
        "./media/video/running.mp4",
        "./media/video/running.mp4",
        "./media/video/running.mp4",
        "./media/video/running.mp4",
        "./media/video/running.mp4",
        "./media/video/running.mp4",
        "./media/video/running.mp4",
        "./media/video/running.mp4",
        "./media/video/running.mp4",
    ],
    on_video_frame=person_detection_app.infer,
    on_prediction=on_prediction,
)

pipeline.start()
pipeline.join()