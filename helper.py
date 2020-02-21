# Helper file containing useful functions and classes for glare detection notebook
import numpy as np

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        for image in tensor:
            for t, m, s in zip(image, self.mean, self.std):
                t.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
        return np.squeeze(tensor)   