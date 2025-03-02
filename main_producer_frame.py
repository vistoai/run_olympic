import cv2
from tqdm import tqdm
import subprocess
import os
import random
import string

def create_random_folder(base_path='media/recording_result_image'):
    folder_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


if __name__ == "__main__":
    script_path = '/home/valid/Documents/Erwin/run_olympic/main_record_subprocess.py'
    subprocess.Popen(['python3', script_path, 'media/recording_result_image'])

    while True:
        cap = cv2.VideoCapture('media/video/sample_1min.mp4')

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames: {total_frames}")

        folder_path = create_random_folder()

        for idx, _ in enumerate(tqdm(range(total_frames), desc="Processing frames")):
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imwrite(os.path.join(folder_path, f'sample_1min_{idx}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        
        with open(os.path.join(folder_path, 'done.txt'), 'w') as f:
            pass
        cap.release()
