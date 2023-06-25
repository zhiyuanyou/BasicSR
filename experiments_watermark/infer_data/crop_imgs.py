import os

import cv2
import glob


def crop(img_path):
    img_name = os.path.basename(img_path)
    video_name = os.path.basename(os.path.split(img_path)[0])
    save_dir = os.path.join("../../datasets/imgs_crop", video_name)
    os.makedirs(save_dir, exist_ok=True)

    img = cv2.imread(img_path)
    h, w, _ = img.shape
    assert h < w
    h_center = int((h - 1) * 0.6493)
    w_center = int((w - 1) * 0.5)
    h1 = h_center - 50
    h2 = h_center + 50
    w1 = w_center - 200
    w2 = w_center + 200
    img_crop = img[h1:h2, w1:w2]
    save_path = os.path.join(save_dir, img_name)
    cv2.imwrite(save_path, img_crop)


if __name__ == "__main__":
    img_paths = glob.glob("../../datasets/video_imgs/*/*.jpg")
    for img_path in img_paths:
        crop(img_path)
