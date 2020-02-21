# Helper file containing useful functions and classes for glare detection notebook
import numpy as np
from sklearn.metrics import confusion_matrix
import torch

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


def performance(model, dataloader, uncertainty = 0.7):
    """
    Performance tester of PyTorch model

    Args:
        model (torchvision.model): Trained Pytorch model, must return log-softmax distribution
        dataloader(iterator): Gives testing data
        output: [confusion, misclassified, uncertain]
        uncertainty (float): bound for image probability to be uncertain

    Returns: 
        numpy array: confusion matrix
        list: false positive image, probability tuple pairs
        list: false negative image, probability tuple pairs
        list: uncertain image, predition, probability tuple triple
    """

    # Create confusion matrix and misclassified images
    conf = np.zeros((2,2))
    fp = []
    fn = []
    uncertain = []
    # Iterate over all images in dataloader
    for images, labels in dataloader:

        #Find probability distribution and get top probability and class
        ps = torch.exp(model(images))
        top_p, top_class = ps.topk(1, dim = 1)

        conf += confusion_matrix(labels.numpy(), top_class.view(*labels.shape).numpy())

        for i in range(len(top_class)):
            if top_class[i] != labels[i]:
                if top_class[i] == 0:
                    fn += [(images[i], top_p[i])]
                else:
                    fp += [(images[i], top_p[i])]
            elif top_p[i] < 0.7:
                uncertain += [(images[i], top_class[i], top_p[i])]


    return conf, fp, fn, uncertain