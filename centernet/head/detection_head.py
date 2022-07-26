from typing import Tuple, List
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_head_nn(in_channels: int, inter_channels: int, out_channels: int):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            inter_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        ),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            inter_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
    )

# TODO add initialize weights

class DetectionHead2D(nn.Module):

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()

        self.offset_head = get_head_nn(
            in_channels,
            in_channels,
            2,
        )
        self.shape_head = get_head_nn(
            in_channels,
            in_channels,
            2,
        )
        self.cls_head = get_head_nn(
            in_channels,
            in_channels,
            num_classes,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        offset_logits = self.offset_head(x).permute(0, 2, 3, 1)
        shape_logits = self.shape_head(x).permute(0, 2, 3, 1)
        cls_logits = self.cls_head(x).permute(0, 2, 3, 1)

        return cls_logits, offset_logits, shape_logits

    def decode(self, cls_logits: Tensor, offset_logits: Tensor, shape_logits: Tensor, score_threshold: float = 0.1, keep_n: int = 100) -> Tensor:
        """Decodes logits and converts to predictions

        Args:
            cls_logits (Tensor): B x FH x FW x C
            offset_logits (Tensor): B x FH x FW x 2
            shape_logits (Tensor): B x FH x FW x 2
            score_threshold (float, optional): score threshold to filter. Defaults to 0.2.
            keep_n (int, optional): number of maxiumum detections per image. Defaults to 100.

        Returns:
            Tensor: (N, 7) where
                0   -> batch id
                1:5 -> bbox as xmin, ymin, xmax, ymax
                5   -> score between 0 to 1
                6   -> class id
        """
        mp = F.max_pool2d(cls_logits.permute(0, 3, 1, 2), 3, stride=1, padding=1)
        keep =  mp.permute(0, 2, 3, 1) == cls_logits
        bs, gy, gx, cats = torch.where(keep)
        wh_half = shape_logits[bs, gy, gx, :] / 2
        offsets = offset_logits[bs, gy, gx, :]
        scores = torch.sigmoid(cls_logits[bs, gy, gx, cats])

        xmin = gx + offsets[:, 0] - wh_half[:, 0]
        ymin = gy + offsets[:, 1] - wh_half[:, 1]
        xmax = gx + offsets[:, 0] + wh_half[:, 0]
        ymax = gy + offsets[:, 1] + wh_half[:, 1]

        preds = torch.stack([bs, xmin, ymin, xmax, ymax, scores, cats], dim=1)

        # filter out some predictions using score
        preds = preds[scores >= score_threshold, :]

        batch_preds: List[torch.Tensor] = []
        for batch_idx in range(cls_logits.shape[0]):
            (pick_n,) = torch.where(batch_idx == preds[:, 0])
            order = preds[pick_n, 5].sort(descending=True)[1]

            batch_preds.append(
                # preds: n, 7
                preds[pick_n, :][order][:keep_n, :]
            )

        batch_preds = torch.cat(batch_preds, dim=0)
        # batch_preds: N x 7
        batch_preds[:, 1:5] *= 4 # TODO change hardcoded stride
        return batch_preds