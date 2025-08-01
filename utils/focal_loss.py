import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        # alpha can be a scalar or tensor of size num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        probs = F.softmax(inputs, dim=1)
        p_t = torch.gather(probs, 1, targets.unsqueeze(1)).squeeze(1)
        
        # Apply class weights if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (int, float)):
                alpha_t = self.alpha
            else:
                # Move alpha to the same device as targets
                alpha_device = self.alpha.to(targets.device)
                alpha_t = alpha_device[targets]
        else:
            alpha_t = 1.0
            
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
