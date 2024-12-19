import torch
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure as ssim


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        
    def forward(self, output, target):
        return 1 - ssim(output, target, data_range=1.0)


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')  # 不進行內部reduction，保留逐元素損失
    
    def forward(self, output, target, weight):
        # 計算每個像素的 MSE loss
        loss = self.mse(output, target)
        # 應用權重
        weighted_loss = loss * weight
        # 對損失進行求和並平均
        return weighted_loss.mean()

