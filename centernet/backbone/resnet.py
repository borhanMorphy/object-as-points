from typing import List
import math
from torch import Tensor
import torch.nn as nn
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, load_state_dict_from_url, model_urls
from .block import DCN


def fill_upsample_layer_weights(up):
    # ref: https://github.com/xingyizhou/CenterNet/blob/2b7692c377c6686fb35e473dac2de6105eed62c6/src/lib/models/networks/resnet_dcn.py#L110
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 


class ResNet18Plain(ResNet):
    def __init__(self) -> None:
        super().__init__(
            BasicBlock,
            [2, 2, 2, 2],
            norm_layer=nn.BatchNorm2d
        )

        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 128, 64],
            [4, 4, 4],
        )

        # drop last 2 layers
        delattr(self, "avgpool")
        delattr(self, "fc")

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers: int, num_filters: List[int], num_kernels: List[int]):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        inplanes = 512
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            cnn = nn.Conv2d(
                inplanes,
                planes,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            )

            upsample_layer = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)

            fill_upsample_layer_weights(upsample_layer)

            layers += [
                cnn,
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                upsample_layer,
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
            ]
            inplanes = planes

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return self.deconv_layers(x)

    @property
    def out_features(self) -> int:
        # TODO
        return 64

    @classmethod
    def from_pretrained(cls):
        backbone = cls()
        state_dict = load_state_dict_from_url(model_urls["resnet18"], progress=True)
        backbone.load_state_dict(state_dict, strict=False)
        return backbone


# TODO add initialize weights

class ResNet18DCN(ResNet18Plain):

    def _make_deconv_layer(self, num_layers: int, num_filters: List[int], num_kernels: List[int]):
        # ref: https://github.com/xingyizhou/CenterNet/blob/2b7692c377c6686fb35e473dac2de6105eed62c6/src/lib/models/networks/resnet_dcn.py#L209
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        inplanes = 512
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            dcn = DCN(
                inplanes,
                planes, 
                (3, 3),
            )

            upsample_layer = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)

            fill_upsample_layer_weights(upsample_layer)

            layers += [
                dcn,
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                upsample_layer,
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
            ]
            inplanes = planes

        return nn.Sequential(*layers)