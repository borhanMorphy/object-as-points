from typing import Tuple

from torchvision.ops import DeformConv2d
import torch
import torch.nn as nn
from torch import Tensor

# TODO add initialize weights

class DCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
    ) -> None:
    
        super().__init__()

        self.conv2d_offset = nn.Conv2d(
            in_channels,
            3 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        )
        self.deform_conv2d = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        )
        self.split_sizes  = (2 * kernel_size[0] * kernel_size[1], kernel_size[0] * kernel_size[1])

        self.reset_offset()

    def reset_offset(self):
        self.conv2d_offset.weight.data.zero_()
        self.conv2d_offset.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv2d_offset(x)
        # out: B x (3*kh*kw) x fh x fw
 
        offsets, mask = torch.split(out, self.split_sizes, dim=1)
        # offsets: B x (2*kh*kw) x fh x fw
        # mask: B x (kh*kw) x fh x fw

        mask = torch.sigmoid(mask)

        return self.deform_conv2d(x, offsets, mask=mask)


