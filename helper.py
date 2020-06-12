# Helper file containing useful functions and classes for glare detection notebook
import numpy as np
import matplotlib.pyplot as plt
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
        tensor = tensor.clone()
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
    model.eval()
    # Iterate over all images in dataloader
    for images, labels in dataloader:

        #Find probability distribution and get top probability and class
        ps = torch.exp(model(images))
        top_p, top_class = ps.topk(1, dim = 1)

        conf += confusion_matrix(labels.numpy(), top_class.view(*labels.shape).numpy(), labels = [0,1])
        # print(confusion_matrix(labels.numpy(), top_class.view(*labels.shape).numpy(), labels = [0,1]))

        for i in range(len(top_class)):
            if labels[i].item() == 1:
                continue
            print(top_class[i].item(), top_p[i].item())
            # print(top_class[i] == labels[i], top_p[i])
            if top_class[i] != labels[i]:
                if top_class[i] == 0:
                    fn += [(images[i], top_p[i].item())]
                else:
                    fp += [(images[i], top_p[i].item())]
            elif top_p[i] < 0.7:
                uncertain += [(images[i], top_class[i].item(), top_p[i].item())]


    return conf, fp, fn, uncertain



def display(fp, fn, uncertain, normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), certainty = 0.7):
    """
    Displays images from the output of the performance funciton

    Returns
        matlplotlib subplots of images
    """
    
    keys = {1: 'flare', 0: 'good'}
    width = max([len(fp), len(fn), len(uncertain)])
    f, ax = plt.subplots(3, width + 1, figsize = (width * 5, 9))

    ax[0,0].text(0, 0.5, f'{len(fp)} false positives:')
    ax[1,0].text(0, 0.5, f'{len(fn)} false negatives:')
    ax[2,0].text(0, 0.5, f'{len(uncertain)} instances of certainty < {certainty}:')

    ax[0,0].set_axis_off()
    ax[1,0].set_axis_off()
    ax[2,0].set_axis_off()
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    for i in range(width):
        ax[0,i+1].set_axis_off()
        ax[1,i+1].set_axis_off()
        ax[2,i+1].set_axis_off()

        if len(fp) > i:
            ax[0,i+1].imshow(unorm(fp[i][0]).numpy().transpose((1, 2, 0)))
            ax[0,i+1].set_title(f'Predicted {fp[i][1]:.3f} chance flare')

        if len(fn) > i:
            ax[1,i+1].imshow(unorm(fn[i][0]).numpy().transpose((1, 2, 0)))
            ax[1,i+1].set_title(f'Predicted {fn[i][1]:.3f} chance good')

        if len(uncertain) > i:
            ax[2,i+1].imshow(unorm(uncertain[i][0].clone()).numpy().transpose((1, 2, 0)))
            ax[2,i+1].set_title(f'{uncertain[i][2]:.3f} certainty {keys[uncertain[i][1]]}')

    plt.show()