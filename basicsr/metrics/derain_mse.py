import numpy as np

from basicsr.metrics.metric_util import reorder_image
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_mse_derain(img, img2, mask, target, binarize_mask, type_norm, input_order='HWC', **kwargs):
    """
    Args:
        img (ndarray): Result images with range [0, 255].
        img2 (ndarray): GT images with range [0, 255].
        mask (ndarray): Rain masks with range [0, 255].
        target (str): Choices: [bg, rain]
        binarize_mask (bool): If True, binarize mask, else, rescale mask to [0,1].
        type_norm (str): Choices: [bg, img, None].
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.

    Returns:
        float: MSE result.
    """

    assert img.shape == img2.shape == mask.shape, (
        f'Image shapes are different: {img.shape}, {img2.shape}, {mask.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    mask = reorder_image(mask, input_order=input_order)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    mask = mask.astype(np.float64)

    if binarize_mask:
        mask[mask != 0] = 1
    else:
        mask = mask / 255

    if target == "bg":
        mask = 1 - mask

    if type_norm == "bg":
        # use background as norm
        norm = np.sum((img - img2)**2 * (1 - mask)) / np.sum(1 - mask)
    elif type_norm == "img":
        # use image as norm
        norm = np.mean((img - img2)**2)
    else:
        norm = 1
    mse = np.sum((img - img2)**2 * mask) / np.sum(mask)
    return mse / norm
