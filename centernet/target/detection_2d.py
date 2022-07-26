from typing import Tuple
import math

import numpy as np

from ..utils import gaussian_2d, gaussian_radius


class Detection2DTarget():
    def __init__(self, output_stride: int = 4, num_classes: int = 80) -> None:
        self._output_stride = output_stride
        self._num_classes = num_classes

    def build_targets(self, img_h: int, img_w: int, bboxes: np.ndarray, label_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fh = img_h // self._output_stride
        fw = img_w // self._output_stride

        if bboxes.shape[0] == 0:
            return (
                np.zeros((fh, fw, self._num_classes), dtype=np.float32),
                np.zeros((fh, fw, 2), dtype=np.float32),
                np.zeros((fh, fw, 2), dtype=np.float32),
            )

        cls_targets = self.build_cls_targets(img_h, img_w, bboxes, label_ids)
        offset_targets = self.build_offset_targets(img_h, img_w, bboxes)
        shape_targets = self.build_shape_targets(img_h, img_w, bboxes)

        return cls_targets, offset_targets, shape_targets

    def build_cls_targets(self, img_h: int, img_w: int, bboxes: np.ndarray, label_ids: np.ndarray) -> np.ndarray:
        f_bboxes = bboxes / self._output_stride
        fh = img_h // self._output_stride
        fw = img_w // self._output_stride

        heatmap = np.zeros((fh, fw, self._num_classes), dtype=np.float32)
        for (x1, y1, x2, y2), label_id in zip(f_bboxes, label_ids):
            radius = gaussian_radius((math.ceil(y2 - y1), math.ceil(x2 - x1)))
            radius = max(0, int(radius))

            grid_x = math.floor((x1 + x2) / 2)
            grid_y = math.floor((y1 + y2) / 2)

            diameter = radius * 2 + 1

            sigma = diameter / 6

            kernel = gaussian_2d(diameter, sigma)
            # kernel: diameter x diameter
            # heatmap: fh x fw x nC

            top, bottom = max(grid_y - radius, 0), min(grid_y + radius + 1, fh)
            left, right = max(grid_x - radius, 0), min(grid_x + radius + 1, fw)

            k_left, k_right = max(radius - grid_x, 0), min(fw - (grid_x - radius), diameter)
            k_top, k_bottom = max(radius - grid_y, 0), min(fh - (grid_y - radius), diameter)

            clipped_heatmap = heatmap[top: bottom, left: right, label_id]
            clipped_kernel = kernel[k_top: k_bottom, k_left: k_right]

            # in place assigment for heatmap
            np.maximum(clipped_heatmap, clipped_kernel, out=clipped_heatmap)

        return heatmap

    def build_offset_targets(self, img_h: int, img_w: int, bboxes: np.ndarray) -> np.ndarray:
        fh = img_h // self._output_stride
        fw = img_w // self._output_stride

        offsets = np.zeros((fh, fw, 2), dtype=np.float32)
        f_bboxes = bboxes / self._output_stride

        box_centers = (f_bboxes[:, :2] + f_bboxes[:, 2:]) / 2

        grids = (box_centers).astype(np.int32)

        grid_x, grid_y = grids.T

        offsets[grid_y, grid_x, :] = box_centers - grids

        return offsets

    def build_shape_targets(self, img_h: int, img_w: int, bboxes: np.ndarray) -> np.ndarray:
        fh = img_h // self._output_stride
        fw = img_w // self._output_stride
        shapes = np.zeros((fh, fw, 2), dtype=np.float32)

        f_bboxes = bboxes / self._output_stride
        grid_x, grid_y = ((f_bboxes[:, :2] + f_bboxes[:, 2:]) / 2).astype(np.int32).T
        shapes[grid_y, grid_x, :] = f_bboxes[:, 2:] - f_bboxes[:, :2]

        return shapes
