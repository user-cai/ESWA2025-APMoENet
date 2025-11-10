import torch
import torch.nn as nn
import torch.nn.functional as F


class FreExpert(nn.Module):
    def __init__(self, nc):
        super(FreExpert, self).__init__()
        self.fpre = nn.Conv2d(nc, nc, 1, 1, 0)
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.process2 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(self.fpre(x), norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        pha = self.process2(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')

        return x_out+x


class SpaExpert(nn.Module):
    # MoE 空间专家块

    def __init__(self, nc):
        super(SpaExpert, self).__init__()

        # 深度可分离卷积
        self.depthwise = nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1, groups=nc, bias=False)
        self.bn1 = nn.BatchNorm2d(nc)  # 增加 BN 以提高稳定性
        self.bn2 = nn.BatchNorm2d(nc)  # 增加 BN 以提高稳定性
        self.pointwise = nn.Conv2d(nc, nc, kernel_size=1, stride=1, bias=False)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        # 深度可分离卷积
        x_dw = self.depthwise(x)
        x_dw = self.bn1(x_dw)  # 归一化
        x_dw = self.act(x_dw)  # 立即激活
        # 逐点卷积
        x_pw = self.pointwise(x_dw)
        x_pw = self.bn2(x_pw)  # 归一化
        x_pw = self.act(x_pw)  # 逐点卷积后再激活

        return x + x_pw


class LaplaceGatingNetwork(nn.Module):
    """ Laplace 门控网络 """

    def __init__(self, in_channels, num_experts, k=3):
        super(LaplaceGatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.k = k  # 选择 Top-K 个专家
        self.frconv = nn.Conv2d(3, in_channels ,kernel_size=1, stride=1, bias=False)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, F1, F2, fr):
        """
        F1, F2: 形状为 (batch, c, h, w)
        """
        # 全局平均池化到 (batch, in_channels, 1, 1) 再 squeeze 成 (batch, in_channels)
        x1 = self.global_avg_pool(F1).squeeze(-1).squeeze(-1)  # (batch, in_channels)
        x2 = self.global_avg_pool(F2).squeeze(-1).squeeze(-1)
        fr = self.frconv(fr)                # (batch, in_channels, H, W)
        fr = self.global_avg_pool(fr).squeeze(-1).squeeze(-1)  # (batch, in_channels)
        dist1 = torch.abs(fr - x1) # (batch, c)
        dist2 = torch.abs(fr - x2) # (batch, c)
        dist1 = - dist1
        dist2 = - dist2
        # 选择 Top-K 专家
        topk_values1, topk_indices1 = torch.topk(dist1, self.k, dim=1)
        topk_values2, topk_indices2 = torch.topk(dist2, self.k, dim=1)
        topk_weights1 = F.softmax(topk_values1, dim=1)
        topk_weights2 = F.softmax(topk_values2, dim=1)

        return topk_weights1, topk_indices1, topk_weights2, topk_indices2


class FuseMoE(nn.Module):
    """ 用于融合 F1 和 F2 """

    def __init__(self, in_channels, num_experts, k=2):
        super(FuseMoE, self).__init__()

        self.gating_network = LaplaceGatingNetwork(in_channels, num_experts, k)
        self.k = k
        self.conv_last = nn.Conv2d(3 * 2, 3, kernel_size=3, stride=1, padding=1)
        self.attention_conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),  # 提取局部信息
            nn.ReLU(inplace=True),  # 激活
            nn.Conv2d(8, 3, kernel_size=1, bias=False),  # 输出 3 通道权重
        )

    def forward(self, F1, F2, fr):
        batch_size, _, H, W = F1.shape

        topk_weights1, topk_indices1, topk_weights2, topk_indices2 = self.gating_network(F1, F2, fr)
        # 选取对应的通道
        F1_selected = torch.gather(F1, 1,
                                   topk_indices1.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W))  # (batch, 3, h, w)
        F2_selected = torch.gather(F2, 1,
                                   topk_indices2.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W))  # (batch, 3, h, w)

        # 扩展权重形状以便广播
        topk_weights1 = topk_weights1.unsqueeze(-1).unsqueeze(-1)  # (batch, 3, 1, 1)
        topk_weights2 = topk_weights2.unsqueeze(-1).unsqueeze(-1)  # (batch, 3, 1, 1)

        # 乘上对应的权重
        F1_out = F1_selected * topk_weights1  # (batch, 3, h, w)
        F2_out = F2_selected * topk_weights2  # (batch, 3, h, w)
        F_fused = self.conv_last(torch.cat([F1_out, F2_out], dim=1))
        F_fused = self.attention_conv(F_fused)
        F_att = torch.sigmoid(F_fused)
        return F_att # (batch, 3, h, w)


