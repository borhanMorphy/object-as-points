import torch
import torch.nn as nn
from torchvision.ops.focal_loss import sigmoid_focal_loss

class NegWeightedPixelWiseFocalLoss(nn.Module):

    def __init__(
        self,
        beta: float = 4.0,
        alpha: float = -1, # ignore
        gamma: float = 2.0,
        reduction: str = "none",
    ) -> None:
        super().__init__()
        self.beta = beta
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        masked_target = target.long().float()
        loss_tmp = sigmoid_focal_loss(
            input, masked_target, alpha=self.alpha, gamma=self.gamma, reduction="none"
        )
        neg_weights = torch.pow(torch.ones_like(loss_tmp) - target, self.beta)
        neg_weights[masked_target == 1] = 1.0
        loss_tmp *= neg_weights

        if self.reduction == "none":
            loss = loss_tmp
        elif self.reduction == "mean":
            loss = torch.mean(loss_tmp)
        elif self.reduction == "sum":
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")
        return loss
