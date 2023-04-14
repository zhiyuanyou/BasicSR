import numpy as np
import os.path as osp
import torch.utils.data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_lmdb
from basicsr.data.derain_util import RainGenerator
from basicsr.data.transforms import augment
from basicsr.data.transforms_derain import paired_random_crop_with_mask, paired_resize_with_mask
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, imwrite, scandir
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class DIV2KDerainDataset(data.Dataset):
    """Dataset specified for Deraining.

    Read GT images and then generate degraded LQ images on-the-fly.
    """

    def __init__(self, opt):
        super(DIV2KDerainDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.add_rain = opt.get("add_rain", None)
        self.vis_lq = opt.get("vis_lq", None)

        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.gt_folder, line.split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        self.rain_prob = opt["rain_prob"]
        self.rain_types = opt["rain_types"]
        self.beta = opt["beta"]
        self.rain_generator = RainGenerator(self.rain_types, self.beta)
        logger = get_root_logger()
        logger.info(f"generate rain with prob: {self.rain_prob}")
        logger.info(f"rain_generator with types: {self.rain_types}, beta: {self.beta}")

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        # Load gt images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=False)
        h, w, _ = img_gt.shape

        # add rain
        img_rain = img_gt.copy()
        rain = np.zeros((h, w, 3))
        if np.random.uniform() < self.rain_prob and self.add_rain:
            rain, img_rain = self.rain_generator(img_gt)

        # resize
        resize = self.opt["resize"]
        img_gt, rain, img_rain = paired_resize_with_mask(img_gt, rain, img_rain, resize)
        if self.opt['phase'] == 'train':
            crop_size = self.opt['crop_size']
            # random crop
            img_gt, rain, img_rain = paired_random_crop_with_mask(img_gt, rain, img_rain, crop_size, scale, gt_path)
            # flip, rotation
            use_flip = self.opt.get("use_flip", False)
            use_rot = self.opt.get("use_rot", False)
            img_gt, rain, img_rain = augment([img_gt, rain, img_rain], use_flip, use_rot)

        if self.vis_lq["flag"]:
            img_name = osp.splitext(osp.basename(gt_path))[0]
            save_img_path = osp.join(self.opt["vis_path"], self.opt["name"], f'{img_name}_{self.vis_lq["suffix"]}.png')
            imwrite(img_rain, save_img_path)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, rain, img_rain = img2tensor([img_gt, rain, img_rain], bgr2rgb=True, float32=True)
        img_gt, rain, img_rain = img_gt / 255., rain / 255., img_rain / 255.

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_rain, self.mean, self.std, inplace=True)

        return {'gt': img_gt, 'rain': rain, 'lq': img_rain, 'gt_path': gt_path, 'lq_path': gt_path}

    def __len__(self):
        return len(self.paths)
