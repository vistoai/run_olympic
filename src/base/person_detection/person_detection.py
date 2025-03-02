import os
import json
from dataclasses import dataclass, field
from typing import Any, List

import torch
import supervision as sv
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=False, kw_only=False, match_args=False, slots=False)
class PersonDetection:
    # Parameters
    set_config: dict = field(default_factory=dict)
    model_config: dict = field(default_factory=dict)
    inference_config: dict = field(default_factory=dict)
    tracker_config: dict = field(default_factory=dict)

    # Variables
    _model: YOLO = field(init=False, repr=False)
    _tracker: sv.ByteTrack = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Model and Tracker
        self._model = YOLO(**self.model_config)
        self._tracker = sv.ByteTrack(**self.tracker_config)
        
    @staticmethod
    def preprocess(data):
        return data

    @staticmethod
    def postprocess(data):
        return data

    def process(self, frame, raw_result: bool = False) -> dict:
        with torch.no_grad():
            results = self._model.predict(
                frame,
               **self.inference_config
            )[0]

        # Compile result to supervision
        detections = sv.Detections.from_ultralytics(results)
        detections = self._tracker.update_with_detections(detections)

        if raw_result:
            return detections
        
        detections = self.postprocess(detections)
        return detections