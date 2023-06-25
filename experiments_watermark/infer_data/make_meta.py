import os
import glob

if __name__ == "__main__":
    img_paths = sorted(glob.glob("../../datasets/imgs_crop/*/*.jpg"))
    img_names = []
    for img_path in img_paths:
        img_dir, img_name = os.path.split(img_path)
        video_name = os.path.basename(img_dir)
        img_name = f"{video_name}/{img_name}"
        img_names.append(img_name)

    with open("infer.txt", "w") as fw:
        fw.write("\n".join(img_names))
