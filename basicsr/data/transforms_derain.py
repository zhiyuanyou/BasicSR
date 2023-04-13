import numpy as np
import random
import torch
from PIL import Image


def paired_random_crop_with_mask(img_gts, mask_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        mask_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT masks.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(mask_gts, list):
        mask_gts = [mask_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
        mask_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in mask_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
        mask_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in mask_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(mask_gts) == 1:
        mask_gts = mask_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, mask_gts, img_lqs


def paired_resize_with_mask(img_gt, mask_gt, img_lq, resize, resize_small=False):
    """
    Args:
        img_gt (ndarray): GT image.
        mask_gt (ndarray): GT mask.
        img_lq (ndarray): LQ image.
        resize (int): size for resize.
        resize_small (bool, optional): If True, image size smaller than resize will be resized. Defaults to False.

    Returns:
        ndarray: GT image, GT mask, LQ image.
    """

    assert img_gt.shape == mask_gt.shape == img_lq.shape, "size of gt, mask, lq must be the same!"
    h, w, _ = img_gt.shape
    if max(h, w) >= resize or resize_small:
        ratio = resize / max(h, w)
        h_new, w_new = round(h * ratio), round(w * ratio)

        img_gt = np.array(Image.fromarray(img_gt).resize((w_new, h_new), Image.Resampling.BICUBIC))
        mask_gt = np.array(Image.fromarray(mask_gt).resize((w_new, h_new), Image.Resampling.BICUBIC))
        img_lq = np.array(Image.fromarray(img_lq).resize((w_new, h_new), Image.Resampling.BICUBIC))

    return img_gt, mask_gt, img_lq
