import os
import sys
import json
from typing import Any, List
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), './src')))

from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame

from base.config import object_detection_config
from app.person_detection_roboflow_app import PersonDetectionRoboflowApp

def on_prediction(prediction: dict, video_frame: VideoFrame) -> None:
    print(f"Prediction: {prediction}")  

person_detection_app = PersonDetectionRoboflowApp(
    config=object_detection_config
)

pipeline = InferencePipeline.init_with_custom_logic(
  video_reference="./my_video.mp4",
  on_video_frame=person_detection_app.infer,
  on_prediction=on_prediction,
)

pipeline.start()
pipeline.join()