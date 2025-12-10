import torch
from torch import nn
import torchvision.transforms.v2 as T


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.lvl1_down = DownLevel(in_channels, 64)
        self.lvl2_down = DownLevel(64, 128)
        self.lvl3_down = DownLevel(128, 256)
        self.lvl4_down = DownLevel(256, 512)
        
        self.lvl5 = Embed(512, 1024)
        
        self.lvl4_up = UpLevel(1024, 56)
        self.lvl3_up = UpLevel(512, 104)
        self.lvl2_up = UpLevel(256, 200)
        self.lvl1_up = UpLevel(128, 392)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x, lvl1 = self.lvl1_down(batch)
        x, lvl2 = self.lvl2_down(x)
        x, lvl3 = self.lvl3_down(x)
        x, lvl4 = self.lvl4_down(x)
        
        x = self.lvl5(x)

        x = self.lvl4_up(x, lvl4)
        x = self.lvl3_up(x, lvl3)
        x = self.lvl2_up(x, lvl2)
        x = self.lvl1_up(x, lvl1)

        x = self.out(x)
        x = T.functional.resize(x, batch.shape[2:])

        return x
    
class DownLevel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.embed = Embed(in_channels, out_channels)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(x)
        return self.downsample(x), x


class UpLevel(nn.Module):
    def __init__(self, in_channels: int, size: int):
        super().__init__()
        self.size = size
        self.up_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.embed = Embed(in_channels, in_channels // 2)

    def forward(self, x: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x)
        x = self.upsample(x)

        crop = T.functional.center_crop(cat, x.shape[2:])
        x = torch.cat((x, crop), dim=1)

        x = self.embed(x)
        return x

class Embed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_relu1 = ConvRelu(in_channels, out_channels)
        self.conv_relu2 = ConvRelu(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_relu1(x)
        x = self.conv_relu2(x)
        return x


class ConvRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        return x