import cv2
import numpy as np
from tqdm import tqdm
import os
import sys
import time

def create_video_flat(image_folder, video_name):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort(key=lambda x: int(x.split('.')[0]))

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for image in tqdm(images, desc="Creating video"):
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()


def create_video_dynamic(image_paths, video_path):
    frame = cv2.imread(os.path.join(image_paths[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for image_path in tqdm(image_paths, desc="Creating video"):
        video.write(cv2.imread(image_path))

    video.release()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_video.py <image_folder> <video_name>")
        sys.exit(1)
    
    main_folder = sys.argv[1]
    record_type = sys.argv[2]

    if record_type == "flat":
        while True:
            folder_paths = [os.path.join(main_folder, d) for d in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, d))]
            for folder_path in folder_paths:
                folder_name = os.path.basename(folder_path)
                
                # Check if done.txt exists
                done_file_path = os.path.join(folder_path, "done.txt")
                if os.path.exists(done_file_path):
                    create_video_flat(folder_path, folder_name + ".mp4")
            
            # Sleep for a specified interval before checking again
            time.sleep(5)  # Check every 60 seconds
    elif record_type == "dynamic":
        # Amount of second that you want the each video being made
        record_per_second = 10

        while True:
            folder_paths = [os.path.join(main_folder, d) for d in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, d))]
            for folder_path in folder_paths:
                folder_name = os.path.basename(folder_path)
                
                # Get all images
                images = [img for img in os.listdir(folder_path) if img.endswith(".jpg")]
                images.sort(key=lambda x: int(x.split('.')[0]))
                image_numbers = [int(img.split('.')[0]) for img in images]

                oldest = min(image_numbers)
                newest = max(image_numbers)

                gap = newest - oldest
                number_of_videos = int(gap // record_per_second)

                for i in range(number_of_videos):
                    start_number = oldest + i * record_per_second
                    end_number = start_number + record_per_second

                    selected_images = [img for img in images if start_number <= int(img.split('.')[0]) < end_number]
                    image_paths = [os.path.join(folder_path, img) for img in selected_images]

                    if image_paths:
                        create_video_dynamic(image_paths, f"{folder_name}_{i}.mp4")
                        
                # Delete the selected images
                for img_path in image_paths:
                    os.remove(img_path)
            
            # Sleep for a specified interval before checking again
            time.sleep(5)  # Check every 60 seconds
