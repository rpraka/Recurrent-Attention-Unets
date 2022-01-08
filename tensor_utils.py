def crop_tensor(x1, x0):
    """
    Crops x1 to the size of x0. x1 must be larger than x0.
    """
    x0_h, x0_w = x0.shape[-2:]  # pull last two dimensions for HxW
    x1_h, x1_w = x1.shape[-2:]
    assert x1_h >= x0_h, "x1 tensor is shorter than x0 tensor"
    assert x1_w >= x0_w, "x1 tensor is narrower than x0 tensor"

    diff_h = (x1_h - x0_h)
    offset = 1 if diff_h % 2 == 1 else 0
    diff_h = diff_h//2

    # crop x1 tensor
    x1 = x1[..., diff_h + offset:x1_h-diff_h, diff_h +
            offset: x1_h-diff_h]  # reassign only last two dims

    return x1
