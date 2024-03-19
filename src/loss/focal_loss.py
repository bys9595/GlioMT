import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

class BinaryFocalLoss(torch.nn.modules.loss._Loss):
    """
    Inherits from torch.nn.modules.loss._Loss. Finds the binary focal loss between each element
    in the input and target tensors.

    Parameters
    -----------
        gamma: float (optional)
            power to raise (1-pt) to when computing focal loss. Default is 2
        reduction: string (optional)
            "sum", "mean", or "none". If sum, the output will be summed, if mean, the output will
                be averaged, if none, no reduction will be applied. Default is mean

    Attributes
    -----------
        gamma: float (optional)
            focusing parameter -- power to raise (1-pt) to when computing focal loss. Default is 2
        reduction: string (optional)
            "sum", "mean", or "none". If sum, the output will be summed, if mean, the output will
                be averaged, if none, no reduction will be applied. Default is mean
    """
    def __init__(self, gamma=2, reduction='mean'):
        if reduction not in ("sum", "mean", "none"):
            raise AttributeError("Invalid reduction type. Please use 'mean', 'sum', or 'none'.")
        super().__init__(None, None, reduction)
        self.gamma = gamma
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, input_tensor, target):
        """
        Compute binary focal loss for an input prediction map and target mask.

        Arguments
        ----------
            input_tensor: torch.Tensor
                input prediction map
            target: torch.Tensor
                target mask

        Returns
        --------
            loss_tensor: torch.Tensor
                binary focal loss, summed, averaged, or raw depending on self.reduction
        """
        
        if input_tensor.min() < 0:
            input_tensor = F.sigmoid(input_tensor)

        # Warn that if sizes don't match errors may occur    
        if not target.size() == input_tensor.size():
            warnings.warn(
                f"Using a target size ({target.size()}) that is different to the input size"\
                "({input_tensor.size()}). \n This will likely lead to incorrect results"\
                "due to broadcasting.\n Please ensure they have the same size.",
                stacklevel=2,
            )
        #Broadcast to get sizes/shapes to match
        input_tensor, target = torch.broadcast_tensors(input_tensor, target)
        assert input_tensor.shape == target.shape, "Input and target tensor shapes don't match"

        #Vectorized computation of binary focal loss
        pt_tensor = (target == 0)*(1-input_tensor) + target*input_tensor
        pt_tensor = torch.clamp(pt_tensor, min=self.eps, max=1.0) #Avoid vanishing gradient
        loss_tensor = -(1-pt_tensor)**self.gamma*torch.log(pt_tensor)

        #Apply reduction
        if self.reduction =='none':
            return loss_tensor
        if self.reduction=='mean':
            return torch.mean(loss_tensor)
        #If not none or mean, sum
        return torch.sum(loss_tensor)
    
    
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-CE_loss)
        F_loss = at*(1-pt)**self.gamma * CE_loss
        return F_loss.mean()

'''
Implementation of Focal Loss.
Reference:
[1]  T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, Focal loss for dense object detection.
     arXiv preprint arXiv:1708.02002, 2017.
'''

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, ignore_index=-100, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        if self.ignore_index >= 0:
            index = torch.nonzero(target.squeeze() != self.ignore_index).squeeze()
            input = input[index, :]
            target = target[index, :]

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()