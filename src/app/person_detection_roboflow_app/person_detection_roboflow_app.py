from dataclasses import dataclass, field
from typing import Any, List

from ultralytics import YOLO
from dotenv import load_dotenv
from inference.core.interfaces.camera.entities import VideoFrame

from base.person_detection import PersonDetection

load_dotenv()

@dataclass(frozen=False, kw_only=False, match_args=False, slots=False)
class PersonDetectionRoboflowApp:
    # Parameters
    config: dict = field(default_factory=dict)

    # Variables
    _model: YOLO = field(init=False, repr=False)
    def __post_init__(self) -> None:
        self._model = PersonDetection(self.config)

    def infer(self, video_frames: List[VideoFrame]) -> List[Any]: 
        # result must be returned as list of elements representing model prediction for single frame with order unchanged.
        predictions = []
        for frame in [v.image for v in video_frames]:
            prediction = self._model.process(
                frame=frame,
                raw_result=False
            )
            predictions.append(prediction)
        
        return predictions