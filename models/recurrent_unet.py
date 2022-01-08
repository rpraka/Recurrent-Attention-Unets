import torch
from torch import nn
from torchviz import make_dot

from dice_losses import DiceLoss
from utils.tensor_utils import crop_tensor


class RecurrentBlock(nn.Module):
    """"Implements RCNN and RRCNN Blocks with Conv2d -> BN -> ReLU"""

    def __init__(self, in_channels, out_channels, t, padding=0, bn=True, residual=True):
        """"
        Reccurent Convolutional Block initialization.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            t (int): t recurrent convolutions will be performed for a total of t+1 convolutions.
            padding (int or tuple): padding amount on each side of input.
            bn (bool): use batch norm after convolutions.
            residual (bool): enable residual connections in recurrent blocks.

        """
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.residual = residual

        self.convF = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding=0)

        convR_modules = [nn.Conv2d(out_channels, out_channels,
                                   kernel_size=3, bias=True, padding=padding), nn.ReLU(inplace=True)]

        if bn:
            convR_modules.insert(-1, nn.BatchNorm2d(out_channels))

        self.convR = nn.Sequential(*convR_modules)

    def forward(self, x):
        x = self.convF(x)  # match x shape with recurrent block's output
        y = 0

        for _ in range(self.t+1):
            y = self.convR(x + y)

        if self.residual:  # RRCNN Block
            y = y + x

        return y


class RecurrentEncoder(nn.Module):
    """Implements encoder downsampling (double conv -> pool)."""

    def __init__(self, in_channels, out_channels, t, padding=0, bn=True, residual=True):
        """"
        Recurrent Encoder initialization.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            t (int): t recurrent convolutions will be performed for a total of t+1 convolutions.
            padding (int or tuple): padding amount on each side of input.
            bn (bool): use batch norm after convolutions.
            residual (bool): enable residual connections in recurrent blocks.

        """
        super(RecurrentEncoder, self).__init__()
        self.encode = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            RecurrentBlock(in_channels, out_channels,
                           t=t, padding=padding, bn=bn, residual=residual),
            RecurrentBlock(out_channels, out_channels,
                           t=t, padding=padding, bn=bn, residual=residual))

    def forward(self, x):
        return self.encode(x)


class RecurrentDecoder(nn.Module):
    """Implements recurrent decoder upsampling (double conv -> transposed conv)"""

    def __init__(self, in_channels, out_channels, t=2, padding=0, bn=True, residual=True):
        """"
        Recurrent Decoder block intialization.
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            t (int): t recurrent convolutions will be performed for a total of t+1 convolutions.
            padding (int or tuple): padding amount on each side of input.
            bn (bool): use batch norm after convolutions.
            residual (bool): enable residual connections in recurrent blocks.

        """
        super(RecurrentDecoder, self).__init__()

        """"
        Note: There seems to be inconsistency in the original paper, where upconvolution kernel size is 3x3,
        but maxpooling uses a 2x2 filter, causing a size mismatch.
        """
        self.upsample = nn.Sequential(nn.ConvTranspose2d(
            in_channels, in_channels//2, kernel_size=2, stride=2))
        self.conv = nn.Sequential(
            RecurrentBlock(
                in_channels, out_channels, t=t, padding=padding, bn=bn, residual=residual),
            RecurrentBlock(
                out_channels, out_channels, t=t, padding=padding, bn=bn, residual=residual))

    def forward(self, x0, x1):
        """"
        Reccurent Decoder forward pass.

        Args:
            x0 (torch.Tensor): a decoder side tensor to be concatenated with x1.
            x1 (torch.Tensor): an encoder tensor to be cropped and concatenated with x0.
        """
        x0 = self.upsample(x0)
        x1 = crop_tensor(x1, x0)  # crop x1 to size of x0
        x2 = torch.cat([x1, x0], dim=1)  # assign result to x2
        x2 = self.conv(x2)
        return x2


class RecurrentUNet(nn.Module):

    def __init__(self, in_channels=3,  t=2, n_features=64, n_segmap_features=2, padding=0, bn=True, residual=True):
        """"
        RUNet initialization. Can be configured to create RUNet or R2UNet via the residual param.

        Args:
            in_channels (int): number of input channels from the data sample.
            t (int): t recurrent convolutions will be performed for a total of t+1 convolutions.
            n_features (int): number of features the input is scaled to by the first encoder block.
            n_segmap_features (int): number of features output in the final segmentation map.
            padding (int or tuple): padding amount on each side of input.
            bn (bool): use batch norm after convolutions.
            residual (bool): enable residual connections in recurrent blocks.
        """
        super(RecurrentUNet, self).__init__()

        self.enc1 = RecurrentBlock(
            in_channels, n_features, t=t, padding=padding, bn=bn, residual=residual)
        self.enc2 = RecurrentEncoder(
            n_features, n_features*2, t=t, padding=padding, bn=bn, residual=residual)
        self.enc3 = RecurrentEncoder(
            n_features*2, n_features*4, t=t, padding=padding, bn=bn, residual=residual)
        self.enc4 = RecurrentEncoder(
            n_features*4, n_features*8, t=t, padding=padding, bn=bn, residual=residual)
        self.enc5 = RecurrentEncoder(
            n_features*8, n_features*16, t=t, padding=padding, bn=bn, residual=residual)

        self.dec1 = RecurrentDecoder(
            n_features*16, n_features * 8, t=t, padding=padding, bn=bn, residual=residual)
        self.dec2 = RecurrentDecoder(
            n_features*8, n_features * 4, t=t, padding=padding, bn=bn, residual=residual)
        self.dec3 = RecurrentDecoder(
            n_features*4, n_features * 2, t=t, padding=padding, bn=bn, residual=residual)
        self.dec4 = RecurrentDecoder(
            n_features*2, n_features, t=t, padding=padding, bn=bn, residual=residual)
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
    model = RecurrentUNet(3, 2, 32, 2, padding=1, bn=False, residual=True)
    print(model)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print('out', y.shape)
    print(y.min())
    # make_dot(y, params=dict(list(model.named_parameters()))
    #          ).render("unet_diagram", format="png")
