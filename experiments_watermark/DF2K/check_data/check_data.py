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
    print(f"Num data: {len(lq_paths)}")

    paths = [f"{lq} {gt}" for lq, gt in zip(lq_paths, gt_paths)]
    random.shuffle(paths)

    p_train = 0.9
    num_train = int(len(paths) * p_train)
    paths_train = paths[:num_train]
    paths_test = paths[num_train:]

    print(f"Num train: {len(paths_train)}, Num test: {len(paths_test)}")

    with open("train.txt", "w") as fw:
        fw.write("\n".join(paths_train))
    with open("test.txt", "w") as fw:
        fw.write("\n".join(paths_test))
