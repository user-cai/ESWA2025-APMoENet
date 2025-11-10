import torch
import torch.nn as nn
import torch.nn.functional as F

# Depthwise Separable Convolution Expert(DW-Expert)
class Expert1(nn.Module):
    def __init__(self, nc):
        super(Expert1, self).__init__()
        # 深度可分离卷积：深度卷积 + 逐点卷积
        self.depthwise_conv = nn.Conv2d(nc, nc, kernel_size=3, padding=1, groups=nc)  # 深度卷积
        self.pointwise_conv = nn.Conv2d(nc, nc, kernel_size=1)  # 逐点卷积
        self.bn = nn.BatchNorm2d(nc)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        out = sum([self.depthwise_conv(f) for f in features])  # 深度卷积
        out = self.pointwise_conv(out)  # 逐点卷积
        out = self.bn(out)
        out = self.relu(out)
        return out

# Grouped Convolution Expert(G-Expert),
class Expert2(nn.Module):
    def __init__(self, nc):
        super(Expert2, self).__init__()

        self.reduce_conv = nn.Conv2d(nc * 5, nc // 2, kernel_size=1, bias=False)  # 先降维
        self.conv = nn.Conv2d(nc // 2, nc, kernel_size=3, padding=1, groups=4, bias=False)  # 分组卷积
        self.act = nn.ReLU(inplace=True)

    def forward(self, features):
        x = torch.cat(features, dim=1)  # (B, 5*nc, H, W)
        x = self.reduce_conv(x)  # (B, reduced_nc, H, W)
        x = self.conv(x)  # (B, nc, H, W)
        return self.act(x)

# SE Attention Expert(SE-Expert)
class Expert3(nn.Module):
    def __init__(self, nc, reduction=4):
        super(Expert3, self).__init__()
        reduced_channels = max(nc // reduction, 8)  # 避免通道数过小

        # 先降维
        self.reduce_conv = nn.Conv2d(nc * 5, nc, kernel_size=1, bias=False)

        # SE模块
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nc, reduced_channels, kernel_size=1, bias=False),  # 进一步降维
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, nc, kernel_size=1, bias=False),  # 升维回 `nc`
            nn.Sigmoid()
        )

    def forward(self, features):
        x = torch.cat(features, dim=1)  # (B, 5*nc, H, W)
        x = self.reduce_conv(x)  # 降维到 (B, nc, H, W)
        att = self.se(x)  # (B, nc, 1, 1)
        return x * att  # 注意力加权，输出 (B, nc, H, W)


# Multi-scale Expert(Ms-Expert)
class Expert4(nn.Module):
    def __init__(self, nc):
        super(Expert4, self).__init__()
        reduced_nc = 16  # 降维
        self.reduce_conv = nn.Conv2d(nc * 5, reduced_nc, kernel_size=1, bias=False)  # 先降维
        self.dwconv1 = nn.Conv2d(reduced_nc, reduced_nc, kernel_size=3, stride=2, padding=1, groups=reduced_nc, bias=False)  # 深度卷积代替池化
        self.dwconv2 = nn.Conv2d(reduced_nc, reduced_nc, kernel_size=3, stride=4, padding=1, groups=reduced_nc, bias=False)  # 深度卷积代替更大池化
        self.conv = nn.Conv2d(reduced_nc, nc, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(nc)
        self.act = nn.ReLU(inplace=True)

    def forward(self, features):
        x = torch.cat(features, dim=1)  # (B, 5*nc, H, W)
        x = self.reduce_conv(x)  # (B, reduced_nc, H, W)
        x1 = self.dwconv1(x)  # (B, reduced_nc, H/2, W/2)
        x2 = self.dwconv2(x)  # (B, reduced_nc, H/4, W/4)
        x = x + F.interpolate(x1, size=x.shape[2:], mode="bilinear", align_corners=False) \
              + F.interpolate(x2, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = self.conv(x)  # (B, nc, H, W)
        x = self.bn(x)
        return self.act(x)


class EfficientGatingNetwork(nn.Module):
    def __init__(self, nc, num_experts=4):
        super(EfficientGatingNetwork, self).__init__()
        self.num_experts = num_experts

        # 通道注意力分支
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # GAP
            nn.Conv2d(nc, nc // 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc // 2, num_experts, kernel_size=1, bias=False)  # 输出 (b, num_experts, 1, 1)
        )

        # 空间注意力分支
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(nc, num_experts, kernel_size=3, padding=1),  # 深度可分离卷积
            nn.ReLU(inplace=True),
            nn.Conv2d(num_experts, num_experts, kernel_size=1, bias=False)  # 维度匹配
        )
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        # 最终融合权重
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        channel_att = self.channel_fc(x).view(x.shape[0], self.num_experts)  # (b, num_experts)
        spatial_att = F.adaptive_avg_pool2d(self.spatial_conv(x), 1).view(x.shape[0], self.num_experts)  # (b, num_experts)
        # 归一化
        channel_att = channel_att / (channel_att.abs().mean(dim=1, keepdim=True) + 1e-6)
        spatial_att = spatial_att / (spatial_att.abs().mean(dim=1, keepdim=True) + 1e-6)
        # 归一化权重
        weights = self.softmax(self.alpha * channel_att + self.beta * spatial_att)  # (b, num_experts)

        return weights


class Stage2MOE(nn.Module):
    def __init__(self, nc):
        super(Stage2MOE, self).__init__()
        self.experts = nn.ModuleList([
            Expert1(nc),
            Expert2(nc),
            Expert3(nc),
            Expert4(nc)
        ])
        self.gate = EfficientGatingNetwork(nc, num_experts=4)

    def forward(self, features, final_feat):

        weights = self.gate(final_feat)  # (b, 4)
        expert_outputs = torch.stack([expert(features) for expert in self.experts], dim=1)  # (b, 4, nc, h, w)
        F_MoE = torch.einsum('bk,bkcHW->bcHW', weights, expert_outputs)  # `einsum` 计算更稳定
        return final_feat + F_MoE


# 测试 MoE 模型
if __name__ == "__main__":
    batch_size = 5
    in_channels = 64
    num_experts = 4
    k = 2
    h = 64
    w = 64
    model = Stage2MOE(in_channels)
    features = [torch.randn(batch_size, in_channels, h, w) for _ in range(5)]
    F2 = torch.randn(batch_size, in_channels, h, w)  # 频域特征
    output = model(features, F2)
    print(output.shape)