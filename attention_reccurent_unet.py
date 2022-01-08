import torch
from torch import nn
from attention_unet import AttentionGate
from reccurent_unet import RecurrentBlock, RecurrentDecoder, RecurrentEncoder


class AttnRecUnet(nn.Module):

    def __init__(self, in_channels=3,  t=2, n_features=64, n_segmap_features=2, padding=0, bn=True, residual=True, inter_channels=-1):
        """"
        Attention Recurrent UNet initialization. Can be configured to create RUNet or R2UNet via the residual param.

        Args:
            in_channels (int): number of input channels from the data sample.
            t (int): t recurrent convolutions will be performed for a total of t+1 convolutions.
            n_features (int): number of features the input is scaled to by the first encoder block.
            n_segmap_features (int): number of features output in the final segmentation map.
            padding (int or tuple): padding amount on each side of input.
            bn (bool): use batch norm after convolutions.
            residual (bool): enable residual connections in recurrent blocks.
            inter_channels (int): number of intermediate channels used in attention gates. 
            Passing -1 will pull number of channels in x.
        """
        super(AttnRecUnet, self).__init__()

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

        self.att1 = AttentionGate(n_features*16, inter_channels=inter_channels)
        self.dec1 = RecurrentDecoder(
            n_features*16, n_features * 8, t=t, padding=padding, bn=bn, residual=residual)

        self.att2 = AttentionGate(n_features*8, inter_channels=inter_channels)
        self.dec2 = RecurrentDecoder(
            n_features*8, n_features * 4, t=t, padding=padding, bn=bn, residual=residual)

        self.att3 = AttentionGate(n_features*4, inter_channels=inter_channels)
        self.dec3 = RecurrentDecoder(
            n_features*4, n_features * 2, t=t, padding=padding, bn=bn, residual=residual)

        self.att4 = AttentionGate(n_features*2, inter_channels=inter_channels)
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
    model = AttnRecUnet(3, 2, 32, 2, padding=1, bn=False,
                        residual=True, inter_channels=-1)
    print(model)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print('out', y.shape)
    print(y.min())
    # make_dot(y, params=dict(list(model.named_parameters()))
    #          ).render("unet_diagram", format="png")
