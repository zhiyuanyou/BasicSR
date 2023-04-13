import numpy as np
import os

from basicsr.data import build_dataloader, build_dataset
from basicsr.utils import imwrite, tensor2img
from basicsr.utils.options import yaml_load

if __name__ == "__main__":
    num_sample = 10
    temp_test_dir = "./temp_test_dir/"
    os.makedirs(temp_test_dir, exist_ok=True)

    opt = yaml_load("./config.yml")
    for phase, dataset_opt in opt['datasets'].items():
        dataset_opt['phase'] = phase
        dataset_opt['scale'] = opt['scale']
        if phase == 'train':
            train_set = build_dataset(dataset_opt)
            train_loader = build_dataloader(train_set, dataset_opt, seed=opt['manual_seed'])
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(val_set, dataset_opt, seed=opt['manual_seed'])

    for idx, batch in enumerate(train_loader):
        gt_path = batch["gt_path"]
        gt, rain, img_rain = batch["gt"], batch["rain"], batch["img_rain"]
        gt, rain, img_rain = tensor2img([gt, rain, img_rain], rgb2bgr=True, out_type=np.uint8, min_max=(0, 1))
        imwrite(gt, os.path.join(temp_test_dir, f"{idx + 1}_gt.jpg"))
        imwrite(rain, os.path.join(temp_test_dir, f"{idx + 1}_rain.jpg"))
        imwrite(img_rain, os.path.join(temp_test_dir, f"{idx + 1}_img_rain.jpg"))
        print(f"Succeed: batch {idx + 1}")

        if idx + 1 >= num_sample:
            break
