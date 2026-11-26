import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class GlobalAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(GlobalAttentionBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_sa = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, inputs):
        x = self.avgpool(inputs)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        C_A = x * inputs
        avg_out = torch.mean(C_A, dim=1, keepdim=True)
        max_out, _ = torch.max(C_A, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv_sa(x)
        x = torch.sigmoid(x)
        S_A = x * C_A
        return S_A

class CategoryAttentionBlock(nn.Module):
    def __init__(self, in_channels, classes, k):
        super().__init__()
        self.classes = classes
        self.k = k
        self.conv = nn.Conv2d(in_channels, k * classes, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(k * classes)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.convM = nn.Conv2d(classes, 1, kernel_size=7, padding=3, bias=False)  # optional

    def forward(self, x):
        B, C, H, W = x.shape
        F1 = torch.relu(self.bn(self.conv(x)))  # (B, k*classes, H, W)

        # Global max pooling
        pooled = self.global_max_pool(F1).view(B, self.classes, self.k)  # (B, classes, k)
        S = torch.softmax(torch.mean(pooled, dim=-1), dim=-1)  # (B, classes)

        # Reshape and category mean
        x_cat = F1.view(B, self.classes, self.k, H, W).mean(dim=2)  # (B, classes, H, W)
        x_cat = x_cat * S.unsqueeze(-1).unsqueeze(-1)  # class weight broadcast (B, classes, H, W)

        # Spatial mask
        M = torch.sigmoid(self.convM(x_cat))  # (B, 1, H, W)

        # Semantic enhancement
        out = x * M + x
        return out

class classer(nn.Module):
    def __init__(self, in_channel = 2048,num_class = 2):
        super(classer, self).__init__()
        self.GA = GlobalAttentionBlock(in_channels=in_channel)
        # self.CA = CategoryAttentionBlock(in_channels=in_channel, classes=num_class, k=6) #6
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(in_channel,num_class)

    def forward(self, x):

        x = x.permute(0,3,1,2)
        x = self.GA(x)
        # x = self.CA(x)
        x = self.gap(x).squeeze()
        x = self.linear(x)

        return x