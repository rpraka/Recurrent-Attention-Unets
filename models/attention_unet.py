import torch
from torch import nn
from torchviz import make_dot
from models.unet import ConvBlock, Decoder, Encoder


class AttentionGate(nn.Module):
    """Implements Attention Gate skip connection"""

    def __init__(self, in_channels, inter_channels):
        """"
        Attention Gate intialization.
        Args:
            in_channels (int): number of input channels from decoder side (g).
            out_channels (int): number of output channels.
        """
        super(AttentionGate, self).__init__()
        if inter_channels == -1:
            inter_channels = in_channels*2

        self.Wg = nn.Conv2d(in_channels, inter_channels,
                            kernel_size=1, stride=1, padding=0)
        self.Wx = nn.Conv2d(in_channels//2, inter_channels,
                            kernel_size=1, stride=(2, 2), padding=0)

        self.comp_sig = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, out_channels=1,
                      kernel_size=1, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        """"
        Attention Gate forward pass.

        Args:
            g (torch.Tensor): a decoder side tensor to be passed as the gating signal.
            x (torch.Tensor): an encoder tensor to be cropped and attention gated with g.
        """

        assert x.shape[-2:] == tuple([2*i for i in g.shape[-2:]]
                                     ), "x height/width are not double those of g, padding=1 may resolve this."

        theta_x = self.Wx(x)
        phi_g = self.Wg(g)
        xg = theta_x + phi_g
        sig = self.comp_sig(xg)
        a = nn.Upsample(  # original paper uses trilinear upsampling for 3D image tensors
            size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)(sig)
        y = a * x

        return y


class AttentionUNet(nn.Module):

    def __init__(self, in_channels=3, inter_channels=-1,  n_features=64, n_segmap_features=2, padding=0, bn=True):
        """"
        Attention UNet initialization.

        Args:
            in_channels (int): number of input channels from the data sample.
            inter_channels (int): number of intermediate channels used in attention gates. 
            Passing -1 will pull number of channels in x.
            n_features (int): number of features the input is scaled to by the first encoder block.
            n_segmap_features (int): number of features output in the final segmentation map.
            padding (int or tuple): padding amount on each side of input.
            bn (bool): use batch norm after convolutions.
        """
        super(AttentionUNet, self).__init__()

        self.enc1 = ConvBlock(in_channels, n_features, padding=padding, bn=bn)
        self.enc2 = Encoder(n_features, n_features*2, padding=padding, bn=bn)
        self.enc3 = Encoder(n_features*2, n_features*4, padding=padding, bn=bn)
        self.enc4 = Encoder(n_features*4, n_features*8, padding=padding, bn=bn)
        self.enc5 = Encoder(n_features*8, n_features *
                            16, padding=padding, bn=bn)

        self.att1 = AttentionGate(n_features*16, inter_channels=inter_channels)
        self.dec1 = Decoder(n_features*16, n_features *
                            8, padding=padding, bn=bn)

        self.att2 = AttentionGate(n_features*8, inter_channels=inter_channels)
        self.dec2 = Decoder(n_features*8, n_features*4, padding=padding, bn=bn)

        self.att3 = AttentionGate(n_features*4, inter_channels=inter_channels)
        self.dec3 = Decoder(n_features*4, n_features*2, padding=padding, bn=bn)

        self.att4 = AttentionGate(n_features*2, inter_channels=inter_channels)
        self.dec4 = Decoder(n_features*2, n_features, padding=padding, bn=bn)
        self.conv1x1 = nn.Conv2d(n_features, n_segmap_features, kernel_size=1)

    def forward(self, x):
        assert x.shape[-2] == x.shape[-1], "Pass a square input tensor"
        ye1 = self.enc1(x)
        ye2 = self.enc2(ye1)
        ye3 = self.enc3(ye2)
        ye4 = self.enc4(ye3)
        ye5 = self.enc5(ye4)

        a1 = self.att1(ye5, ye4)
        yd1 = self.dec1(ye5, a1)

        a2 = self.att2(yd1, ye3)
        yd2 = self.dec2(yd1, a2)

        a3 = self.att3(yd2, ye2)
        yd3 = self.dec3(yd2, a3)

        a4 = self.att4(yd3, ye1)
        yd4 = self.dec4(yd3, a4)

        segmap = self.conv1x1(yd4)
        segmap = torch.sigmoid(segmap)
        return segmap


if __name__ == "__main__":
    model = AttentionUNet(3, -1, 64, 1, padding=1, bn=True)
    print(model)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print('out', y.shape)
    print(y.min())
