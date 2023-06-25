import os

import cv2
from PIL import Image


def imgs2video(img_dir_pred, img_dir, media_path, fps):
    img_names = os.listdir(img_dir)
    img_names.sort(key=lambda n: int(n[:-4]))
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    image = Image.open(os.path.join(img_dir, img_names[0]))
    media_writer = cv2.VideoWriter(media_path, fourcc, fps, image.size)
    for img_name in img_names:
        img_path_pred = os.path.join(img_dir_pred, img_name.replace(".jpg", "_res.png"))
        img_pred = cv2.imread(img_path_pred)
        img = cv2.imread(os.path.join(img_dir, img_name))
        h, w, _ = img.shape
        assert h < w
        h_center = int((h - 1) * 0.6493)
        w_center = int((w - 1) * 0.5)
        h1 = h_center - 50
        h2 = h_center + 50
        w1 = w_center - 200
        w2 = w_center + 200
        img[h1:h2, w1:w2] = img_pred
        media_writer.write(img)
    media_writer.release()
    print(f"{img_dir_pred} End")


if __name__ == "__main__":
    fps = 25
    lq_dir = "../../datasets/video_imgs"
    save_dir = "../../experiments/001_MSRResNet_DF2K_120k_B32_GPU4/infer_iter5k/videos_pred"
    pred_dir = "../../experiments/001_MSRResNet_DF2K_120k_B32_GPU4/infer_iter5k/visualization/VIDEO"
    os.makedirs(save_dir, exist_ok=True)
    video_names = os.listdir(lq_dir)
    for video_name in video_names:
        img_dir_pred = os.path.join(pred_dir, video_name)
        img_dir = os.path.join(lq_dir, video_name)
        media_path = os.path.join(save_dir, f"{video_name}.mp4")
        imgs2video(img_dir_pred, img_dir, media_path, fps=fps)
