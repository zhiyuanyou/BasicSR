import cv2
import numpy as np
import os

from basicsr.data.derain_util import RainGenerator

if __name__ == "__main__":
    num_img = 1
    save_dir = "./temp_test_dir"
    os.makedirs(save_dir, exist_ok=True)
    img = cv2.imread("../data/div2k_0390.png")

    for rain_type in ["small", "medium", "large", "random"]:
        for idx in range(num_img):
            beta = np.random.uniform(0.6, 1)
            beta = np.round(beta, 2)
            rain_generator = RainGenerator([rain_type], [beta, beta])
            rain, img_rain = rain_generator(img)
            cv2.imwrite(os.path.join(save_dir, "orig.png"), img)
            cv2.imwrite(
                os.path.join(save_dir, f"{rain_type}_{beta}_{idx}_img.png"),
                img_rain,
            )
            cv2.imwrite(os.path.join(save_dir, f"{rain_type}_{beta}_{idx}_rain.png"), rain)
            print(rain_type, beta, idx)
