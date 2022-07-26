from typing import Optional, Union, Tuple
from enum import Enum
import random

import numpy as np
from cv2 import cv2
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F
from albumentations.augmentations.bbox_utils import denormalize_bbox, normalize_bbox


def gaussian_radius(det_size, min_overlap=0.7):
    # ref: https://github.com/princeton-vl/CornerNet/blob/e5c39a31a8abef5841976c8eab18da86d6ee5f9a/sample/utils.py#L27
    height, width = det_size
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2

    return min(r1, r2, r3)

def gaussian_1d(kernel_size: int, sigma: float) -> np.ndarray:
    # make sure kernel size is odd
    assert kernel_size % 2 == 1, "use odd kernel size"

    kernel = np.arange(kernel_size, dtype=np.float32) - (kernel_size - 1) / 2

    return np.exp(-0.5 * (kernel / sigma) ** 2)

def gaussian_2d(kernel_size: int, sigma: float) -> np.ndarray:
    k = gaussian_1d(kernel_size, sigma).reshape(kernel_size, 1)
    return k @ k.T


class PadIfNeededv2(DualTransform):

    class PositionType(Enum):
        CENTER = "center"
        TOP_LEFT = "top_left"
        TOP_RIGHT = "top_right"
        BOTTOM_LEFT = "bottom_left"
        BOTTOM_RIGHT = "bottom_right"
        RANDOM = "random"

    def __init__(
        self,
        min_height: Optional[int] = 1024,
        min_width: Optional[int] = 1024,
        pad_height_divisor: Optional[int] = None,
        pad_width_divisor: Optional[int] = None,
        position: Union[PositionType, str] = PositionType.CENTER,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=1.0,
    ):
        if (min_height is None) == (pad_height_divisor is None):
            raise ValueError("Only one of 'min_height' and 'pad_height_divisor' parameters must be set")

        if (min_width is None) == (pad_width_divisor is None):
            raise ValueError("Only one of 'min_width' and 'pad_width_divisor' parameters must be set")

        super(PadIfNeededv2, self).__init__(always_apply, p)
        self.min_height = min_height
        self.min_width = min_width
        self.pad_width_divisor = pad_width_divisor
        self.pad_height_divisor = pad_height_divisor
        self.position = PadIfNeededv2.PositionType(position)
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def update_params(self, params, **kwargs):
        params = super(PadIfNeededv2, self).update_params(params, **kwargs)
        rows = params["rows"]
        cols = params["cols"]

        target_w = (rows | 31) + 1
        target_h = (cols | 31) + 1

        h_pad_top = (target_h - cols) // 2 + (target_h - cols) % 2
        h_pad_bottom = (target_h - cols) // 2
        w_pad_left = (target_w - rows) // 2 + (target_w - rows) % 2
        w_pad_right = (target_w - rows) // 2

        params.update(
            {
                "pad_top": h_pad_top,
                "pad_bottom": h_pad_bottom,
                "pad_left": w_pad_left,
                "pad_right": w_pad_right,
            }
        )
        return params

    def apply(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return F.pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=self.value,
        )

    def apply_to_mask(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return F.pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=self.mask_value,
        )

    def apply_to_bbox(self, bbox, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, rows=0, cols=0, **params):
        x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)
        bbox = x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top
        return normalize_bbox(bbox, rows + pad_top + pad_bottom, cols + pad_left + pad_right)

    # skipcq: PYL-W0613
    def apply_to_keypoint(self, keypoint, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        x, y, angle, scale = keypoint
        return x + pad_left, y + pad_top, angle, scale

    def get_transform_init_args_names(self):
        return (
            "min_height",
            "min_width",
            "pad_height_divisor",
            "pad_width_divisor",
            "border_mode",
            "value",
            "mask_value",
        )

    def __update_position_params(
        self, h_top: int, h_bottom: int, w_left: int, w_right: int
    ) -> Tuple[int, int, int, int]:
        if self.position == PadIfNeededv2.PositionType.TOP_LEFT:
            h_bottom += h_top
            w_right += w_left
            h_top = 0
            w_left = 0

        elif self.position == PadIfNeededv2.PositionType.TOP_RIGHT:
            h_bottom += h_top
            w_left += w_right
            h_top = 0
            w_right = 0

        elif self.position == PadIfNeededv2.PositionType.BOTTOM_LEFT:
            h_top += h_bottom
            w_right += w_left
            h_bottom = 0
            w_left = 0

        elif self.position == PadIfNeededv2.PositionType.BOTTOM_RIGHT:
            h_top += h_bottom
            w_left += w_right
            h_bottom = 0
            w_right = 0

        elif self.position == PadIfNeededv2.PositionType.RANDOM:
            h_pad = h_top + h_bottom
            w_pad = w_left + w_right
            h_top = random.randint(0, h_pad)
            h_bottom = h_pad - h_top
            w_left = random.randint(0, w_pad)
            w_right = w_pad - w_left

        return h_top, h_bottom, w_left, w_right