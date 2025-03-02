import os
import json
from dataclasses import dataclass, field
from typing import Any, List

import numpy as np
from rich import print
import supervision as sv
from dotenv import load_dotenv
from paddleocr import PaddleOCR

load_dotenv()

@dataclass(frozen=False, kw_only=False, match_args=False, slots=False)
class PaddleOcrDetectionRecognition:
    # Parameters
    model_config: dict = field(default_factory=dict)

    # Variables
    _model: PaddleOCR = field(init=False, repr=False)
    def __post_init__(self) -> None:
        # PaddleOCR
        self._model = PaddleOCR(**self.model_config)

    @staticmethod
    def preprocess(data):
        return data

    @staticmethod
    def postprocess(data):
        return data

    def process(self, frame, raw_result: bool = False) -> dict:
        # OCR
        xyxys = []
        confidences = []
        texts = []
        result = self._model.ocr(frame, cls=True)
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                coordinates, text_and_confidence = line

                x1y1, x2y2, x3y3, x4y4 = coordinates
                text, confidence = text_and_confidence

                xyxys.append(x1y1 + x3y3)
                confidences.append(confidence)
                texts.append(text)

        # Convert to numpy
        xyxys = np.array(xyxys)
        confidences = np.array(confidences)
        class_ids = np.zeros(len(xyxys), dtype=int)
        texts = np.array(texts)
        
        # Create detections based on Supervision
        detections = sv.Detections(
            xyxy=xyxys,
            confidence=confidences,
            class_id=class_ids,   
            data={
                'text': texts
            }
        )

        if raw_result:
            return detections
        
        detections = self.postprocess(detections)
        return detections