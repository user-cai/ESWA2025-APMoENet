import torch
import torch.nn as nn


class AmpAttention(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nc, nc // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc // 4, nc, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        c_att = self.channel_att(x)
        x = x * c_att
        x_max = torch.amax(x, dim=1, keepdim=True)  # (B, 1, H, W)
        x_mean = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        s_att = self.spatial_att(torch.cat([x_max, x_mean], dim=1))  # (B, 2, H, W)
        return x * s_att


class AmpGuidePha(nn.Module):
    def __init__(self, channels=8, k=4):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # g(xamp), shape: (B, C, 1, 1)
        # K 个 1×1 Conv，每个将 C -> C // K
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels // k, kernel_size=1)
            for _ in range(k)
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, xamp):
        pooled = self.global_pool(xamp)  # (B, C, 1, 1)
        expert_feats = [conv(pooled) for conv in self.convs]
        concatenated = torch.cat(expert_feats, dim=1)  # (B, C, 1, 1)
        return self.sigmoid(concatenated)  # (B, C, 1, 1)


class PhaGuideAmp(nn.Module):
    def __init__(self, channels=8, k=4):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # g(xamp), shape: (B, C, 1, 1)
        # K 个 1×1 Conv，每个将 C -> C // K
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels // k, kernel_size=1)
            for _ in range(k)
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, xamp):
        pooled = self.global_pool(xamp)  # (B, C, 1, 1)
        expert_feats = [conv(pooled) for conv in self.convs]
        concatenated = torch.cat(expert_feats, dim=1)  # (B, C, 1, 1)
        return self.sigmoid(concatenated)  # (B, C, 1, 1)


class AmpPhaBlock(nn.Module):
    def __init__(self, nc):
        super(AmpPhaBlock, self).__init__()
        self.pre = nn.Conv2d(nc, nc, 1, 1, 0)
        self.process_pha_vis = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.process_amp_vis = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.process_amp_fr = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        
        self.spa1 = nn.Conv2d(nc, nc, kernel_size=1)
        self.spa2 = nn.Conv2d(nc, nc, kernel_size=3, padding=1)
        self.spa3 = nn.Conv2d(nc, nc, kernel_size=5, padding=2)
        self.spa_conv = nn.Conv2d(nc * 3, nc, 1)
        self.AGP = AmpGuidePha(channels=nc, k=4)
        self.PGA = PhaGuideAmp(channels=nc, k=4)
        self.process_pha_fr = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.sigmoid = torch.nn.Sigmoid()
        self.amp_attention = AmpAttention(nc)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fuse = nn.Conv2d(nc * 2, nc, 1, 1, 0)
        self.fuse_amp = nn.Conv2d(nc * 2, nc, 1, 1, 0)

    def pha_fuse(self, vis, fra):
        features1_flattened = vis.view(vis.size(0), vis.size(1), -1)
        features2_flattened = fra.view(fra.size(0), fra.size(1), -1)
        multiplied = torch.mul(features1_flattened, features2_flattened)
        multiplied_softmax = torch.softmax(multiplied, dim=2)
        multiplied_softmax = multiplied_softmax.view(vis.size(0), vis.size(1),vis.size(2), vis.size(3))
        vis_map = vis * multiplied_softmax + vis
        return vis_map

    def forward(self, vis, fr_amp, fr_pha):
        _, _, H, W = vis.shape
        vis_ori = vis
        vis= torch.fft.rfft2(self.pre(vis), norm='backward')
        mag_vis = torch.abs(vis)
        pha_vis = torch.angle(vis)

        mag_vis = self.process_amp_vis(mag_vis)
        pha_vis = self.process_pha_vis(pha_vis)
        fr_amp = self.process_amp_fr(fr_amp)
        fr_pha = self.process_pha_fr(fr_pha)

        M = self.sigmoid(self.amp_attention(self.fuse_amp(torch.cat((fr_amp, mag_vis), dim=1))))
        pha_out = self.pha_fuse(pha_vis, fr_pha)
        mag_out = mag_vis * M
        pha_out = self.AGP(mag_out) * pha_out + pha_out
        mag_out = self.PGA(pha_out) * mag_out + mag_out

        real = mag_out * torch.cos(pha_out)
        imag = mag_out * torch.sin(pha_out)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        vis_out = x_out + vis_ori
        spa_out1 = self.spa1(vis_ori)
        spa_out2 = self.spa2(vis_ori)
        spa_out3 = self.spa3(vis_ori)
        x_spa_out = self.spa_conv(torch.cat((spa_out1, spa_out2, spa_out3), dim=1))
        vis_out = self.fuse(torch.cat((vis_out, x_spa_out), dim=1))

        return vis_out + vis_ori, fr_amp, fr_pha


# 结构一样，第一块对红外特殊处理
class AmpPhaBlock_1(nn.Module):
    def __init__(self, nc):
        super(AmpPhaBlock_1, self).__init__()
        self.pre = nn.Conv2d(nc, nc, 1, 1, 0)
        self.process_pha_vis = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.process_amp_vis = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.process_amp_fr = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.process_pha_fr = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        
        self.spa1 = nn.Conv2d(nc, nc, kernel_size=1)
        self.spa2 = nn.Conv2d(nc, nc, kernel_size=3, padding=1)
        self.spa3 = nn.Conv2d(nc, nc, kernel_size=5, padding=2)
        self.spa_conv = nn.Conv2d(nc * 3, nc, 1)

        self.fuse = nn.Conv2d(nc * 3, nc, 1)
        self.AGP = AmpGuidePha(channels=nc, k=4)
        self.PGA = PhaGuideAmp(channels=nc, k=4)
        self.sigmoid = torch.nn.Sigmoid()
        self.amp_attention = AmpAttention(nc)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_first_fr1 = nn.Conv2d(3, 8, 1, 1, 0, bias=True)
        self.conv_first_fr2 = nn.Conv2d(3, 8, 1, 1, 0, bias=True)
        self.fuse = nn.Conv2d(nc * 2, nc, 1, 1, 0)
        self.fuse_amp = nn.Conv2d(nc * 2, nc, 1, 1, 0)

    def pha_fuse(self, vis, fra):
        features1_flattened = vis.view(vis.size(0), vis.size(1), -1)
        features2_flattened = fra.view(fra.size(0), fra.size(1), -1)
        multiplied = torch.mul(features1_flattened, features2_flattened)
        multiplied_softmax = torch.softmax(multiplied, dim=2)
        multiplied_softmax = multiplied_softmax.view(vis.size(0), vis.size(1),vis.size(2), vis.size(3))
        vis_map = vis * multiplied_softmax + vis
        return vis_map

    def forward(self, vis, fr):
        _, _, H, W = vis.shape
        vis_ori = vis
        vis= torch.fft.rfft2(self.pre(vis), norm='backward')
        mag_vis = torch.abs(vis)
        pha_vis = torch.angle(vis)
        fr_fft = torch.fft.rfft2(fr, norm='backward')
        fr_amp = torch.abs(fr_fft)
        fr_pha = torch.angle(fr_fft)
        fr_amp = self.conv_first_fr1(fr_amp)
        fr_pha = self.conv_first_fr2(fr_pha)

        mag_vis = self.process_amp_vis(mag_vis)
        pha_vis = self.process_pha_vis(pha_vis)
        fr_amp = self.process_amp_fr(fr_amp)
        fr_pha = self.process_pha_fr(fr_pha)

        M = self.sigmoid(self.amp_attention(self.fuse_amp(torch.cat((fr_amp,mag_vis), dim=1 ))))
        pha_out = self.pha_fuse(pha_vis, fr_pha)
        mag_out = mag_vis * M
        pha_out = self.AGP(mag_out) * pha_out + pha_out
        mag_out = self.PGA(pha_out) * mag_out + mag_out
        real = mag_out * torch.cos(pha_out)
        imag = mag_out * torch.sin(pha_out)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        vis_out = x_out + vis_ori

        spa_out1 = self.spa1(vis_ori)
        spa_out2 = self.spa2(vis_ori)
        spa_out3 = self.spa3(vis_ori)
        x_spa_out = self.spa_conv(torch.cat((spa_out1, spa_out2, spa_out3), dim=1))
        vis_out = self.fuse(torch.cat((vis_out, x_spa_out), dim=1))

        return vis_out + vis_ori, fr_amp, fr_pha


# 通道数减半
class DownBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.compress = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.compress(x)
        return x


class StageOneNet(nn.Module):
    def __init__(self, nc):
        super(StageOneNet, self).__init__()
        self.conv1 = AmpPhaBlock_1(nc)
        self.conv2 = AmpPhaBlock(nc)
        self.conv3 = AmpPhaBlock(nc)
        self.conv4 = AmpPhaBlock(nc)
        self.conv4_after = nn.Conv2d(nc, nc, 1, 1, 0)
        self.conv5 = AmpPhaBlock(nc)
        self.conv5_after = nn.Conv2d(nc, nc, 1, 1, 0)
        self.convout = AmpPhaBlock(nc)
        self.conv_final = nn.Conv2d(nc, 3, 1, 1, 0)

        self.pre_vis = nn.Conv2d(3, nc, 1, 1, 0)
        self.down1 = DownBlock(nc * 2)
        self.down2 = DownBlock(nc * 2)
        self.down3 = DownBlock(nc * 2)

    def forward(self, x, fr):

        x = self.pre_vis(x)
        x1, fr_amp1, fr_pha1 = self.conv1(x, fr)
        x2, fr_amp2, fr_pha2 = self.conv2(x1, fr_amp1, fr_pha1)
        x3, fr_amp3, fr_pha3 = self.conv3(x2, fr_amp2, fr_pha2)
        x4, fr_amp4, fr_pha4 = self.conv4(self.down1(torch.cat((x2, x3), dim=1)), fr_amp3, fr_pha3)
        x4 = self.conv4_after(x4)
        x5, fr_amp5, fr_pha5 = self.conv5(self.down2(torch.cat((x1, x4), dim=1)), fr_amp4, fr_pha4)
        x5 = self.conv5_after(x5)
        xout, fr_amp, fr_pha = self.convout(self.down3(torch.cat((x, x5), dim=1)), fr_amp5, fr_pha5)
        xout = self.conv_final(xout)
        return xout


if __name__ == "__main__":

    # 模拟输入，batch size=2，3通道图像，高宽=64
    b, c, h, w = 2, 1, 64, 64
    x = torch.randn(b, c, h, w)     # 可见光输入
    fr = torch.randn(b, c, h, w)    # 红外图输入
    model = StageOneNet(nc=8)
    output = model(x, fr)
    # 打印输出形状
    print(f"Output shape: {output.shape}")