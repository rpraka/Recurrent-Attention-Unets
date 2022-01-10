from torch import nn


class DiceLoss(nn.Module):
    """
    Differentiable, smoothened loss derived from the Sørensen-Dice coefficient.

    Args:
        smoothing (float): smoothing factor added to numerator and denomintor of dice coeff.
    """

    def __init__(self, smoothing=1.0):
        super(DiceLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, mask):
        assert pred.size() == mask.size()
        inter = (pred*mask)
        inter = inter.sum(dim=(-2, -1))
        numer = 2*inter
        denom = pred.sum(dim=(-2, -1))**2 + mask.sum(dim=(-2, -1))**2
        dice = (numer+self.smoothing)/(denom+self.smoothing)
        dice = dice.mean(dim=1)
        dice = dice.mean(dim=0)
        return 1 - dice


class WeightedDiceLoss(nn.Module):
    """
    Dice loss weighted on samples with tumors present in ground truth masks.

    Args:
        smoothing (float): smoothing factor added to numerator and denomintor of dice coeff.
        gamma (float): weighting factor.
    """

    def __init__(self, smoothing=1.0, gamma=2.0):
        super(WeightedDiceLoss, self).__init__()
        self.smoothing = smoothing
        self.gamma = gamma

    def forward(self, pred, mask):
        assert pred.size() == mask.size()
        g = 1
        inter = (pred*mask)
        inter = inter.sum(dim=(-2, -1))
        numer = 2*inter
        denom = (pred**2).sum(dim=(-2, -1)) + (mask**2).sum(dim=(-2, -1))
        dice = (numer+self.smoothing)/(denom+self.smoothing)
        dice = dice.mean(dim=1)
        dice = dice.mean(dim=0)
        if mask.max() > 0:  # filter for tumor-positive slices
            g = self.gamma
        return 1.0 - (dice**g)
