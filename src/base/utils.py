from datetime import datetime

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

def generate_id_based_on_datetime() -> str:
    # Get the current time with microseconds
    current_time = datetime.now()

    # Format the time to include milliseconds (3 decimal places of microseconds)
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    # Convert the formatted time string to an integer (milliseconds since epoch)
    current_time_milliseconds = int(datetime.strptime(current_time_str, "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1000)
    
    return current_time_milliseconds