import os

import glob
import random

random.seed(131)


def check_names(lq_paths, gt_paths):
    lq_names = [os.path.basename(lq_path) for lq_path in lq_paths]
    gt_names = [os.path.basename(gt_path) for gt_path in gt_paths]
    assert lq_names == gt_names


if __name__ == "__main__":
    lq_paths_1 = sorted(glob.glob("/opt/data/share/xtkong/DF2K_new/mask_demask/*.png"))
    gt_paths_1 = sorted(glob.glob("/opt/data/share/xtkong/DF2K_new/GT_demask/*.png"))
    lq_paths_2 = sorted(glob.glob("/opt/data/share/xtkong/examples/img_mask/*.png"))
    gt_paths_2 = sorted(glob.glob("/opt/data/share/xtkong/examples/img_GT/*.png"))
    print(f"{len(lq_paths_1)} {len(gt_paths_1)} {len(lq_paths_2)} {len(gt_paths_2)}")
    lq_paths = lq_paths_1 + lq_paths_2
    gt_paths = gt_paths_1 + gt_paths_2
    check_names(lq_paths, gt_paths)
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
