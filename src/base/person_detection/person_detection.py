import os
import json
from dataclasses import dataclass, field
from typing import Any, List

import torch
import supervision as sv
from ultralytics import YOLO
from dotenv import load_dotenv
from inference.core.interfaces.camera.entities import VideoFrame

load_dotenv()

@dataclass(frozen=False, kw_only=False, match_args=False, slots=False)
class PersonDetection:
    config: dict = field(default_factory=dict)

    device: object = field(init=False, repr=False)
    _model: YOLO = field(init=False, repr=False)
    _tracker: sv.ByteTrack = field(init=False, repr=False)
    object_annotator: sv.BoxAnnotator = field(init=False, repr=False)
    label_annotator: sv.LabelAnnotator = field(init=False, repr=False) 

    def __post_init__(self) -> None:
        print(torch.cuda.is_available())
        print(f"Device: {self.config['device']}")
        # Setup device variable
        if self.config["device"] == "gpu" and torch.cuda.is_available():
            device_ids = self.config["device_ids"].split(",")
            print(f"Device IDs: {device_ids}")
            if len(device_ids) > 1:
                self.device = []
                for device_id in device_ids:
                    self.device.append(int(device_id))
            elif len(device_ids) == 1:
                self.device = int(device_ids[0])
            else:
                self.device = "cpu"
        else:
            self.device = "cpu"
        
        print(f"Device: {self.device}")

        # Supervision
        self.object_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        # Model and Tracker
        self._model = YOLO(self.config["model"])
        self._tracker = sv.ByteTrack(
            track_activation_threshold=self.config["tracker_activation_threshold"],
            minimum_matching_threshold=self.config["tracker_matching_threshold"],
            minimum_consecutive_frames=self.config["tracker_consecutive_frames"],
        )
        
    @staticmethod
    def preprocess(data: dict) -> None:
        return data

    @staticmethod
    def postprocess(data: dict) -> dict:
        return data

    def process(self, frame, raw_result: bool = False) -> dict:
        with torch.no_grad():
            results = self._model(
                frame,
                conf=self.config['confidence_threshold'],
                iou=self.config['iou_threshold'],
                classes=[0],
                device=self.device,
                half=False,
                agnostic_nms=True,
                stream=True
            )

        temp = None
        for result in results:
            temp = result
        result = temp

        detections = sv.Detections.from_ultralytics(result)
        detections = self._tracker.update_with_detections(detections)
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

        annotated_frame = self.object_annotator.annotate(
            scene=frame.copy(),
            detections=detections,
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame.copy(),
            detections=detections,
            labels=labels,
        )

        if raw_result:
            return detections
        
        detections = self.postprocess(detections)
        return detections