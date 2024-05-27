from typing import *
import torch
from torchvision.transforms.functional import gaussian_blur


class DecayScheduler:
    def __init__(
        self, start_val, end_val, start_step: int, end_step: int, geometric=False
    ):
        assert end_step > start_step
        step_diff = end_step - start_step
        if geometric:
            self.slope = (end_val / (start_val + 1e-14)) ** (1 / step_diff)
        else:
            self.slope = (end_val - start_val) / step_diff
        self.start_step = start_step
        self.end_step = end_step
        self.start_val = start_val
        self.end_val = end_val
        self.geometric = geometric

    def get(self, epoch):
        if epoch < self.start_step:
            return self.start_val
        elif epoch >= self.end_step:
            return self.end_val
        else:
            step = epoch - self.start_step
            if self.geometric:
                return self.start_val * self.slope**step
            else:
                return self.start_val + self.slope * step


def closest_odd_int(x):
    return int((x // 2) * 2 + 1)


def blur_tensors(x: torch.tensor, sigma=0.0):
    if sigma == 0:
        return x
    else:
        kernel_size = closest_odd_int(sigma)
        x = gaussian_blur(x, kernel_size=kernel_size, sigma=sigma)
        return x

def blur_tensor_adaptive(x: torch.tensor, sigma=0.0):
    if sigma == 0:
        return x
    else:
        assert x.ndim >= 2

        ada_sigma = min(x.shape[-2:]) / 100 * sigma
        kernel_size = closest_odd_int(ada_sigma * 6)
        x = gaussian_blur(x, kernel_size=kernel_size, sigma=ada_sigma)
        return x
