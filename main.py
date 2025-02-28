import os
import json
from typing import Any, List

from inference.core.interfaces.camera.entities import VideoFrame
from inference import InferencePipeline


TARGET_DIR = "./my_predictions"
class MyModel:

  def __init__(self, weights_path: str):
    self._model = your_model_loader(weights_path)

  # after v0.9.18  
  def infer(self, video_frames: List[VideoFrame]) -> List[Any]: 
    # result must be returned as list of elements representing model prediction for single frame
    # with order unchanged.
    return self._model([v.image for v in video_frames])

def save_prediction(prediction: dict, video_frame: VideoFrame) -> None:
  with open(os.path.join(TARGET_DIR, f"{video_frame.frame_id}.json")) as f:
    json.dump(prediction, f)

my_model = MyModel("./my_model.pt")

pipeline = InferencePipeline.init_with_custom_logic(
  video_reference="./my_video.mp4",
  on_video_frame=my_model.infer,
  on_prediction=save_prediction,
)

pipeline.start()
pipeline.join()