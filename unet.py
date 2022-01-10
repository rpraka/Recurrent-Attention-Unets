import torch
from torch import nn
from torchviz import make_dot
from utils.tensor_utils import crop_tensor


class ConvBlock(nn.Module):
    """"Implements Conv2d -> BN -> ReLU -> Conv2d -> BN -> ReLU."""

    def __init__(self, in_channels, out_channels, padding=0, bn=True):
        """"   
        ConvBlock initialization.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            padding (int or tuple): padding amount on each side of input.
            bn (bool): use batch norm after convolutions.
        """
        super(ConvBlock, self).__init__()

        if bn:
            # batch norm enabled
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, bias=True, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, bias=True, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, bias=True, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, bias=True, padding=padding),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    """Implements encoder downsampling (double conv -> pool)."""

    def __init__(self, in_channels, out_channels, padding=0, bn=True):
        """"
        Encoder initialization.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            padding (int or tuple): padding amount on each side of input.
            bn (bool): use batch norm after convolutions.
        """
        super(Encoder, self).__init__()
        self.encode = nn.Sequential(nn.MaxPool2d(
            kernel_size=2, stride=2), ConvBlock(in_channels, out_channels, padding=padding, bn=bn))

    def forward(self, x):
        return self.encode(x)


class Decoder(nn.Module):
    """Implements decoder upsampling (double conv -> transposed conv)"""

    def __init__(self, in_channels, out_channels, padding=0, bn=True):
        """"
        Decoder block intialization.   
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            padding (int or tuple): padding amount on each side of input.
            bn (bool): use batch norm after convolutions.
        """
        super(Decoder, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels,
                              padding=padding, bn=bn)

    def forward(self, x0, x1):
        """"   
        Decoder forward pass.

        Args:
            x0 (torch.Tensor): a decoder side tensor to be concatenated with x1.
            x1 (torch.Tensor): an encoder tensor to be cropped and concatenated with x0.
        """
        x0 = self.upsample(x0)
        x1 = crop_tensor(x1, x0)

        x2 = torch.cat([x1, x0], dim=1)  # assign result to x2
        x2 = self.conv(x2)
        return x2


class UNet(nn.Module):

    def __init__(self, in_channels=3,  n_features=64, n_segmap_features=2, padding=0, bn=True):
        """"   
        UNet initialization.

        Args:
            in_channels (int): number of input channels from the data sample.
            n_features (int): number of features the input is scaled to by the first encoder block.
            n_segmap_features (int): number of features output in the final segmentation map.
            padding (int or tuple): padding amount on each side of input.
            bn (bool): use batch norm after convolutions.
        """
        super(UNet, self).__init__()

        self.enc1 = ConvBlock(in_channels, n_features, padding=padding, bn=bn)
        self.enc2 = Encoder(n_features, n_features*2, padding=padding, bn=bn)
        self.enc3 = Encoder(n_features*2, n_features*4, padding=padding, bn=bn)
        self.enc4 = Encoder(n_features*4, n_features*8, padding=padding, bn=bn)
        self.enc5 = Encoder(n_features*8, n_features *
                            16, padding=padding, bn=bn)

        self.dec1 = Decoder(n_features*16, n_features *
                            8, padding=padding, bn=bn)
        self.dec2 = Decoder(n_features*8, n_features*4, padding=padding, bn=bn)
        self.dec3 = Decoder(n_features*4, n_features*2, padding=padding, bn=bn)
        self.dec4 = Decoder(n_features*2, n_features, padding=padding, bn=bn)
        self.conv1x1 = nn.Conv2d(n_features, n_segmap_features, kernel_size=1)

    def forward(self, x):
        assert x.shape[-2] == x.shape[-1], "Pass a square input tensor"
        ye1 = self.enc1(x)
        ye2 = self.enc2(ye1)
        ye3 = self.enc3(ye2)
        ye4 = self.enc4(ye3)
        ye5 = self.enc5(ye4)

        yd1 = self.dec1(ye5, ye4)
        yd2 = self.dec2(yd1, ye3)
        yd3 = self.dec3(yd2, ye2)
        yd4 = self.dec4(yd3, ye1)
        segmap = self.conv1x1(yd4)
        segmap = torch.sigmoid(segmap)
        return segmap


if __name__ == "__main__":
    model = UNet(3, 32, 2, padding=1, bn=False)
    print(model)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print('out', y.shape)
    print(y.min())
