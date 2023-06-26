import os

import glob
import random

random.seed(131)

if __name__ == "__main__":
    lq_paths = sorted(glob.glob("/opt/data/share/xtkong/DF2K/mask_demask/*.png"))
    gt_paths = sorted(glob.glob("/opt/data/share/xtkong/DF2K/GT_demask/*.png"))
    lq_names = [os.path.basename(lq_path) for lq_path in lq_paths]
    gt_names = [os.path.basename(gt_path) for gt_path in gt_paths]
    assert lq_names == gt_names
    print(f"Num data: {len(lq_names)}")

    p_train = 0.9
    num_train = int(len(lq_names) * p_train)
    random.shuffle(lq_names)
    names_train = lq_names[:num_train]
    names_test = lq_names[num_train:]
    print(f"Num train: {len(names_train)}, Num test: {len(names_test)}")

    with open("train.txt", "w") as fw:
        fw.write("\n".join(names_train))
    with open("test.txt", "w") as fw:
        fw.write("\n".join(names_test))
