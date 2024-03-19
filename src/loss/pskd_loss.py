import torch.nn as nn
from torch.nn import functional as F

import torch

class Custom_CrossEntropy_PSKD(nn.Module):
    def __init__(self):
        super(Custom_CrossEntropy_PSKD, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        
    def forward(self,output, targets):
        '''
        Args : 
            inputs : prediction matrix (before softmax ) with shape (batch, num_classes)
            targets : ground truth labels with shape (num_classes)
        '''
        log_probs = self.logsoftmax(output)
        loss = (-targets * log_probs).mean(0).sum()
        return loss
    
class Custom_BCE_PSKD(nn.Module):
    def __init__(self):
        super(Custom_BCE_PSKD, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        
    def forward(self,output, targets):
        '''
        Args : 
            inputs : prediction matrix (before softmax ) with shape (batch, num_classes)
            targets : ground truth labels with shape (num_classes)
        '''
        log_probs = self.logsoftmax(output)
        loss = (-targets * log_probs).mean(0).sum()
        return loss

if __name__ == '__main__':
    # target = torch.Tensor([[0.05, 0.7, 0.25],
    #                        [0.7, 0.1, 0.2]]) # 64 classes, batch size = 10
    target = torch.Tensor([0.7,
                           0.2,
                           0.1]) # 64 classes, batch size = 10
    # output = torch.Tensor([[0.15, 0.6, 0.25],
    #                        [0.5, 0.25, 0.25]])  # A prediction (logit)
    output = torch.Tensor([0.8,
                           0.1,
                           0.1])  # A prediction (logit)
    # pos_weight = torch.ones([2])*2  # All weights are equal to 1
    pos_weight = torch.Tensor([3])   # All weights are equal to 1
    # criterion = Custom_CrossEntropy_PSKD()
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = criterion(output, target)  # -log(sigmoid(1.5))
    
    a = 1