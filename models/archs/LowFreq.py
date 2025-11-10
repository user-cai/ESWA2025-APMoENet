import torch
import torch.nn as nn
from models.archs.ffc import FFCResnetBlock
import kornia
import torch.nn.functional as F


class Stage2Fr(nn.Module):
    def __init__(self, nc=64):
        super(Stage2Fr, self).__init__()
        self.fpre = nn.Conv2d(nc, nc, 1, 1, 0)
        self.conv_first_fr = nn.Conv2d(3, nc, 1, 1, 0, bias=True)
        self.process_pha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.process_fr = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    # Infrared Embedding
    def infra_guide(self, vis_pha, fra_pha):
        features1_flattened = vis_pha.view(vis_pha.size(0), vis_pha.size(1), -1)
        features2_flattened = fra_pha.view(fra_pha.size(0), fra_pha.size(1), -1)
        multiplied = torch.mul(features1_flattened, features2_flattened)
        multiplied_softmax = torch.softmax(multiplied, dim=2)
        multiplied_softmax = multiplied_softmax.view(vis_pha.size(0), vis_pha.size(1),vis_pha.size(2), vis_pha.size(3))
        vis_map = vis_pha * multiplied_softmax + vis_pha
        return vis_map

    def forward(self, x, fr):
        _, _, H, W = x.shape
        fr = F.interpolate(fr, size=(H, W), mode='bilinear', align_corners=False)
        # 红外相位提取
        fr_fft = torch.fft.rfft2(fr, norm='backward')
        pha_fr = torch.angle(fr_fft)
        pha_fr = self.lrelu(self.conv_first_fr(pha_fr))
        pha_fr = self.process_fr(pha_fr)

        # 特征图相位提取
        x_freq = torch.fft.rfft2(self.fpre(x), norm='backward')
        mag_ori = torch.abs(x_freq)
        pha_x = torch.angle(x_freq)
        pha_x = self.process_pha(pha_x)

        pha_refine = self.infra_guide(pha_x, pha_fr)
        real = mag_ori * torch.cos(pha_refine)
        imag = mag_ori * torch.sin(pha_refine)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        x_out= x_out + x
        return x_out
