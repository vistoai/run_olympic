import cv2
import numpy as np
from tqdm import tqdm
import os
import sys
import time

def create_video(image_folder, video_name):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for image in tqdm(images, desc="Creating video"):
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_video.py <image_folder> <video_name>")
        sys.exit(1)
    
    main_folder = sys.argv[1]

    while True:
        folder_paths = [os.path.join(main_folder, d) for d in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, d))]
        for folder_path in folder_paths:
            folder_name = os.path.basename(folder_path)
            
            # Check if done.txt exists
            done_file_path = os.path.join(folder_path, "done.txt")
            if os.path.exists(done_file_path):
                create_video(folder_path, folder_name + ".mp4")
        
        # Sleep for a specified interval before checking again
        time.sleep(5)  # Check every 60 seconds
