from dataclasses import dataclass, field
from typing import Any, List

from ultralytics import YOLO
from dotenv import load_dotenv
from inference.core.interfaces.camera.entities import VideoFrame

from base.person_detection import PersonDetection
from base.paddleocr_detection_recognition import PaddleOcrDetectionRecognition

load_dotenv()

@dataclass(frozen=False, kw_only=False, match_args=False, slots=False)
class RunningOlympicRoboflowApp:
    # Parameters
    person_detection_set_config: dict = field(default_factory=dict)
    person_detection_model_config: dict = field(default_factory=dict)
    person_detection_inference_config: dict = field(default_factory=dict)
    person_detection_tracker_config: dict = field(default_factory=dict)
    paddleocr_detection_recognition_model_config: dict = field(default_factory=dict)

    # Variables
    _person_detection_model: YOLO = field(init=False, repr=False)
    _ocr_model: YOLO = field(init=False, repr=False)
    def __post_init__(self) -> None:
        # Define model
        self._person_detection_model = PersonDetection(
            set_config=self.person_detection_set_config,
            model_config=self.person_detection_model_config,
            inference_config=self.person_detection_inference_config,
            tracker_config=self.person_detection_tracker_config
        )
        self._ocr_model = PaddleOcrDetectionRecognition(
            model_config=self.paddleocr_detection_recognition_model_config
        )

    def infer(self, video_frames: List[VideoFrame]) -> List[Any]: 
        # result must be returned as list of elements representing model prediction for single frame with order unchanged.
        result = []
        for frame in [v.image for v in video_frames]:
            # Person detection
            person_detection_prediction = self._person_detection_model.process(
                frame=frame,
                raw_result=False
            )

            # OCR
            ocr_prediction = self._ocr_model.process(
                frame=frame,
                raw_result=False
            )

            result.append((person_detection_prediction, ocr_prediction))
        return result