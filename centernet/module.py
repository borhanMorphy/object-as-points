from typing import Tuple, List

from cv2 import cv2
import torch
import torch.nn as nn
from torch import Tensor
import albumentations as A

from torchmetrics.functional import average_precision

from .backbone import ResNet18DCN, ResNet18Plain
from .head import DetectionHead2D
from .loss import NegWeightedPixelWiseFocalLoss
from .utils import PadIfNeededv2


class CenterNet2D(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = ResNet18Plain.from_pretrained() # TODO
        self.head = DetectionHead2D(
            self.backbone.out_features,
            num_classes
        )

        self.cls_loss_fn = NegWeightedPixelWiseFocalLoss(beta=4, gamma=2, reduction="none")
        self.offset_loss_fn = nn.L1Loss(reduction="none")
        self.shape_loss_fn = nn.L1Loss(reduction="none")


    def forward(self, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """computes logits with given normalized image batch

        Args:
            batch (Tensor):  (B, 3, H, W) normalized image with [0, 1] value range

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                0 -> cls logits as (B, FH, FW, C)   C: number of classes
                1 -> offset logits as (B, FH, FW, 2) x offset, y offset
                2 -> shape logits as (B, FH, FW, 2) predicted width, height
        """
        features = self.backbone(batch)
        return self.head(features)

    @torch.no_grad()
    def predict(self, batch: Tensor, score_threshold: float = 0.3, keep_n: int = 100) -> Tensor:
        """computes predictions with given normalized image batch

        Args:
            batch (Tensor): (B, 3, H, W) normalized image with [0, 1] value range
            score_threshold (float, optional): minimum score required. Defaults to 0.3.
            keep_n (int, optional): maximum detections per image . Defaults to 100.

        Returns:
            Tensor: (N, 7) where
                0   -> batch id
                1:5 -> bbox as xmin, ymin, xmax, ymax
                5   -> score between 0 to 1
                6   -> class id
        """
        cls_logits, offset_logits, shape_logits = self.forward(batch)

        return self.head.decode(cls_logits, offset_logits, shape_logits, score_threshold=score_threshold, keep_n=keep_n)

    def compute_loss(self, logits, targets):
        cls_logits, offset_logits, shape_logits = logits
        cls_targets, offset_targets, shape_targets = targets

        mask = cls_targets.amax(dim=-1).long().float().unsqueeze(-1)

        N = mask.sum()

        cls_loss = self.cls_loss_fn(cls_logits, cls_targets).sum() / N
        offset_loss = self.offset_loss_fn(offset_logits * mask, offset_targets * mask).sum() / N
        shape_loss = self.shape_loss_fn(shape_logits * mask, shape_targets * mask).sum() / N

        # TODO
        loss = cls_loss + offset_loss + shape_loss*0.1

        return dict(
            loss=loss,
            cls_loss=cls_loss,
            offset_loss=offset_loss,
            shape_loss=shape_loss,
        )

    @torch.no_grad()
    def compute_metrics(self, logits, targets):
        cls_logits, _, shape_logits = logits
        cls_targets, _, shape_targets = targets

        mask = cls_targets.amax(dim=-1).long().float().unsqueeze(-1)

        y_hat = torch.sigmoid(cls_logits.detach()).amax(dim=-1).unsqueeze(-1)
        bs, gy, gx, _ = torch.where(mask == 1)
        nbs, ngy, ngx, _ = torch.where(mask != 1)

        cbs, cgy, cgx, cls_indexes = torch.where(cls_targets == 1)

        cls_scores = torch.sigmoid(cls_logits.detach())

        return dict(
            w_ratio=(shape_logits[bs, gy, gx, 0] / shape_targets[bs, gy, gx, 0]).detach().cpu().mean(),
            h_ratio=(shape_logits[bs, gy, gx, 1] / shape_targets[bs, gy, gx, 1]).detach().cpu().mean(),
            pixelwise_objectness_ap=average_precision(y_hat.flatten(), mask.flatten()).cpu().mean(),
            pos_objectness_abs_err=torch.abs(y_hat[bs, gy, gx] - 1).flatten().cpu().mean(),
            neg_objectness_abs_err=torch.abs(y_hat[nbs, ngy, ngx] - cls_targets[nbs, ngy, ngx]).flatten().cpu().mean(),
            pos_grids_cls_acc=(cls_scores[cbs, cgy, cgx, :].argmax(dim=1) == cls_indexes).cpu().float().mean(),
        )


    def configure_optimizers(self, learning_rate: float = 5e-4, milestones: List[int] = [90, 120], lr_gamma: float = 0.1):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=lr_gamma,
        )

        return optimizer, scheduler

    @staticmethod
    def train_transforms(scale_limit: float = 0.4, target_size: int = 512, min_area: int = 2**2, min_visibility: float = 0.7,
            mean: Tuple[float, float, float] = (0, 0, 0), std: Tuple[float, float, float] = (1, 1, 1),
            max_pixel_value: float = 255.0,
        ):
        return A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=1/2 - 1/4, # stride is 4
                    scale_limit=scale_limit,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    rotate_limit=0,
                    p=1.0,
                ),
                A.PadIfNeeded(
                    min_width=target_size,
                    min_height=target_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.CenterCrop(target_size, target_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=0.5),
                A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value)
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                min_area=min_area,
                min_visibility=min_visibility,
            ),
        )

    @staticmethod
    def transforms(target_size: int = 512, min_area: int = 2**2, min_visibility: float = 0.7,
        mean: Tuple[float, float, float] = (0, 0, 0), std: Tuple[float, float, float] = (1, 1, 1),
        max_pixel_value: float = 255.0,
    ):
        return A.Compose(
            [
                PadIfNeededv2(
                    min_width=25000,
                    min_height=25000,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value)
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                min_area=min_area,
                min_visibility=min_visibility,
            ),
        )
