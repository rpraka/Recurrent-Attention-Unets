from torch import nn


class DiceLoss(nn.Module):
    """
    Differentiable, smoothened loss derived from the Sørensen-Dice coefficient.

    Args:
        smoothing (float): smoothing factor added to numerator and denomintor of dice coeff.
        gamma (float): weighting factor.
    """

    def __init__(self, smoothing=1.0):
        super(DiceLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dice_coeff = (2. * intersection + self.smoothing) / \
            (y_pred.sum() + y_true.sum() + self.smoothing)
        return 1.0 - dice_coeff


class WeightedDiceLoss(nn.Module):
    """
    Dice loss weighted on samples with tumors present in ground truth masks.

    Args:
        smoothing (float): smoothing factor added to numerator and denomintor of dice coeff.
    """

    def __init__(self, smoothing=1.0, gamma=2.0):
        super(WeightedDiceLoss, self).__init__()
        self.smoothing = smoothing
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        g = 1
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dice_coeff = (2. * intersection + self.smoothing) / \
            (y_pred.sum() + y_true.sum() + self.smoothing)
        if y_true.max() > 0:  # filter for tumor-positive slices
            g = self.gamma
        return 1.0 - (dice_coeff**g)
