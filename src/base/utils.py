import numpy as np
import supervision as sv

def filter_paddleocr_coordinate_to_person_detection_coordinate(
    person_detection: sv.Detections,
    paddleocr_detection: sv.Detections
) -> np.ndarray:
    filter_result = []

    # Check if paddleocr coordinates are inside person detection coordinates
    for paddleocr_coordinate in paddleocr_detection.xyxy:
        paddle_x1, paddle_y1, paddle_x2, paddle_y2 = paddleocr_coordinate
        is_inside = False
        for person_detection_coordinate in person_detection.xyxy:
            person_x1, person_y1, person_x2, person_y2 = person_detection_coordinate
            if (paddle_x1 > person_x1 and paddle_x2 < person_x2 and
                paddle_y1 > person_y1 and paddle_y2 < person_y2):
                is_inside = True
                break
        filter_result.append(is_inside)
    
    filter_result = np.array(filter_result, dtype=bool)
    return filter_result