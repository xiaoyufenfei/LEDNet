import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d,self).__init__()

        self.loss = nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, dim=1), targets)






