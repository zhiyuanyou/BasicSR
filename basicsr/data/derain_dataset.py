import numpy as np
import os.path as osp
import torch.utils.data as data
from torchvision.transforms.functional import normalize

from basicsr.data.derain_util import RainGenerator, set_val_seed
from basicsr.data.transforms import augment
from basicsr.data.transforms_derain import paired_random_crop_with_mask, paired_resize
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, imwrite, scandir
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class DerainDataset(data.Dataset):
    """Dataset specified for Deraining.

    Read GT images and then generate degraded LQ images on-the-fly.
    """

    def __init__(self, opt):
        super(DerainDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.mean = opt.get("mean", None)
        self.std = opt.get("std", None)
        self.add_rain_cfg = opt.get("add_rain_cfg", None)
        self.vis_lq = opt.get("vis_lq", None)

        self.gt_folder = opt["dataroot_gt"]
        self.lq_folder = opt.get("dataroot_lq", None)

        if "meta_info_file" in self.opt:
            with open(self.opt["meta_info_file"], "r") as fin:
                self.gt_paths = [osp.join(self.gt_folder, line.strip().split(" ")[0]) for line in fin]
        else:
            self.gt_paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        if self.lq_folder:
            if "meta_info_file" in self.opt:
                with open(self.opt["meta_info_file"], "r") as fin:
                    self.lq_paths = [osp.join(self.lq_folder, line.strip().split(" ")[1]) for line in fin]
            else:
                self.lq_paths = sorted(list(scandir(self.lq_folder, full_path=True)))

        if self.add_rain_cfg:
            assert self.lq_folder is None, "lq_folder is not None, no need add rain"
            self.rain_generator = RainGenerator(self.add_rain_cfg["rain_types"], self.add_rain_cfg["beta"])
            logger = get_root_logger()
            logger.info(f"generate rain with prob: {self.add_rain_cfg['rain_prob']}")
            logger.info(
                f"rain_generator with types: {self.add_rain_cfg['rain_types']}, beta: {self.add_rain_cfg['beta']}")

    def __getitem__(self, index):
        # keep added rain same for different iterations
        if self.opt["phase"] in ["val", "test"]:
            set_val_seed(self.opt["manual_seed"] + index)

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop("type"), **self.io_backend_opt)

        scale = self.opt["scale"]
        # Load gt images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.gt_paths[index]
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=False)

        # Load lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        lq_path = gt_path
        img_lq = img_gt.copy()
        if self.lq_folder:
            lq_path = self.lq_paths[index]
            img_bytes = self.file_client.get(lq_path)
            img_lq = imfrombytes(img_bytes, float32=False)

        # resize
        img_gt, img_lq = paired_resize(img_gt, img_lq, self.opt["resize"])
        h, w, _ = img_gt.shape
        rain = np.zeros((h, w, 3), dtype=np.uint8)

        # add rain
        if self.add_rain_cfg:
            if np.random.uniform() < self.add_rain_cfg["rain_prob"]:
                rain, img_lq = self.rain_generator(img_gt)

        if self.opt["phase"] == "train":
            crop_size = self.opt["crop_size"]
            # random crop
            img_gt, rain, img_lq = paired_random_crop_with_mask(img_gt, rain, img_lq, crop_size, scale, gt_path)
            # flip, rotation
            use_flip = self.opt.get("use_flip", False)
            use_rot = self.opt.get("use_rot", False)
            img_gt, rain, img_lq = augment([img_gt, rain, img_lq], use_flip, use_rot)

        if self.vis_lq:
            img_name = osp.splitext(osp.basename(lq_path))[0]
            save_img_path = osp.join(self.opt["vis_path"], self.opt["name"], f"{img_name}_{self.vis_lq['suffix']}.png")
            imwrite(img_lq, save_img_path)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, rain, img_lq = img2tensor([img_gt, rain, img_lq], bgr2rgb=True, float32=True)
        img_gt, rain, img_lq = img_gt / 255., rain / 255., img_lq / 255.

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_lq, self.mean, self.std, inplace=True)

        return {"gt": img_gt, "mask": rain, "lq": img_lq, "gt_path": gt_path, "lq_path": lq_path}

    def __len__(self):
        return min(self.opt.get("num_img", float("inf")), len(self.gt_paths))
