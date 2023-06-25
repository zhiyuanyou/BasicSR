import os

import cv2
import glob


def video2imgs(video_path, save_dir):
    times = 0
    frame_frequency = 1
    os.makedirs(save_dir, exist_ok=True)
    camera = cv2.VideoCapture(video_path)
    while True:
        times = times + 1
        res, img = camera.read()
        if not res:
            print("not res, not img")
            break
        if times % frame_frequency == 0:
            save_path = os.path.join(save_dir, f"{str(times)}.jpg")
            cv2.imwrite(save_path, img)
    camera.release()
    print(f"{video_path} End")


if __name__ == "__main__":
    video_paths = glob.glob("../../datasets/videos/*.mp4")
    for video_path in video_paths:
        save_dir = os.path.join("../../datasets/video_imgs", os.path.splitext(os.path.basename(video_path))[0])
        video2imgs(video_path, save_dir)
