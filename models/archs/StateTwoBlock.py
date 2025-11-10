import torch
import torch.nn as nn
from models.archs.ffc import FFCResnetBlock
import kornia
from models.archs.Stage2MOE import *
from models.archs.LowFreq import *
# Stage2 Multi-scale-Block
class Multi_Conv_Block(nn.Module):
    def __init__(self, dim, num_head=4, expand_ratio=2):
        super(Multi_Conv_Block, self).__init__()
        self.dim = dim
        self.num_head = num_head
        self.split_groups = self.dim // num_head
        self.conv_reduction = nn.Conv2d(dim, dim//4, 1, 1, bias=True)
        self.leakyrelu = nn.LeakyReLU(0.1,inplace=True)
        for i in range(self.num_head):
            local_conv = nn.Conv2d(dim//4, dim//4, kernel_size=(3+i*2), padding=(1+i), stride=1, groups=dim//num_head)
            setattr(self, f"local_conv_{i + 1}", local_conv)
        self.conv3 = nn.Conv2d(dim, dim, 1, 1, bias=True)
    def forward(self, x):
        x_duc = self.leakyrelu(self.conv_reduction(x))
        for i in range(self.num_head):
            local_conv = getattr(self, f"local_conv_{i+1}")
            x_pro = self.leakyrelu(local_conv(x_duc))
            x_pro = x_pro * torch.sigmoid(x_duc)
            if i == 0:
                s_o = x_pro
            else:
                s_o = torch.cat([s_o, x_pro], 1) # 4*64*192*192
        x = self.leakyrelu(self.conv3(s_o)) + x
        return x

# ===================暂时未使用=====================
class FFT_StateTow(nn.Module):
    def __init__(self, nc):
        super(FFT_StateTow, self).__init__()
        self.fpre = nn.Conv2d(nc, nc, 1, 1, 0)
        self.cat = nn.Conv2d(nc*2, nc, 1,1,0)
        self.process_amp = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.process_pha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.process_fr = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.process_sp = nn.Sequential(
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    # Stage1 Infrared Embedding
    def multiply_and_softmax(self, vis, fra):
        features1_flattened = vis.view(vis.size(0), vis.size(1), -1)
        features2_flattened = fra.view(fra.size(0), fra.size(1), -1)
        multiplied = torch.mul(features1_flattened, features2_flattened)
        multiplied_softmax = torch.softmax(multiplied, dim=2)
        multiplied_softmax = multiplied_softmax.view(vis.size(0), vis.size(1),vis.size(2), vis.size(3))
        vis_map = vis * multiplied_softmax + vis
        return vis_map

    def forward(self, x, fr):
        _, _, H, W = x.shape
        # ******** 空间分支 *********
        x_sp = self.process_sp(x)
        x_out_sp = x + x_sp
        # ******** 傅里叶分支 *********
        x_freq = torch.fft.rfft2(self.fpre(x), norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process_amp(mag)
        pha = self.process_pha(pha)
        # ******** 红外分支 ********* fr是红外的pha分量
        fr = self.process_fr(fr)
        pha = self.multiply_and_softmax(pha, fr)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        x_out_ff = x_out + x
        x_cat = self.cat(torch.cat([x_out_ff,x_out_sp],1))
        return x+x_cat, fr


class AmplitudeAugment(nn.Module):
    def __init__(self, nc):
        super(AmplitudeAugment, self).__init__()
        self.conv0a = nn.Conv2d(3, nc, 1, 1, 0)
        self.conv0b =  FFT_StateTow(nc)
        self.conv1 = FFT_StateTow(nc)
        self.conv2 = FFT_StateTow(nc)
        self.conv3 = FFT_StateTow(nc)
        self.conv4 = nn.Sequential(
            FFT_StateTow_NoFr(nc * 2),
            nn.Conv2d(nc * 2, nc, 1, 1, 0),
        )
        self.conv5 = nn.Sequential(
            FFT_StateTow_NoFr(nc * 2),
            nn.Conv2d(nc * 2, nc, 1, 1, 0),
        )
        self.convout = nn.Sequential(
            FFT_StateTow_NoFr(nc * 2),
            nn.Conv2d(nc * 2, 3, 1, 1, 0),
        )

    def forward(self, x, fr):
        x, fr = self.conv0b(self.conv0a(x), fr)
        x1, fr1 = self.conv1(x, fr)
        x2, fr2 = self.conv2(x1, fr1)
        x3, fr3 = self.conv3(x2, fr2)
        x4 = self.conv4(torch.cat((x2, x3), dim=1))
        x5 = self.conv5(torch.cat((x1, x4), dim=1))
        xout = self.convout(torch.cat((x, x5), dim=1))

        return xout


class FFT_StateTow_NoFr(nn.Module):
    def __init__(self, nc):
        super(FFT_StateTow_NoFr, self).__init__()
        self.fpre = nn.Conv2d(nc, nc, 1, 1, 0)
        self.cat = nn.Conv2d(nc*2, nc, 1,1,0)
        self.process_amp = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.process_pha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.process_sp = nn.Sequential(
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        _, _, H, W = x.shape
        # ******** 空间分支 *********
        x_sp = self.process_sp(x)
        x_out_sp = x + x_sp
        # ******** 傅里叶分支 *********
        x_freq = torch.fft.rfft2(self.fpre(x), norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process_amp(mag)
        pha = self.process_pha(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        x_out_ff = x_out + x
        x_cat = self.cat(torch.cat([x_out_ff,x_out_sp],1))
        return x_cat
# ===================暂时未使用=====================

# Stage2 双分支总框架
class DualGenerator(nn.Module):
    def __init__(self, nf=64):
        super(DualGenerator,self).__init__()
        self.conv_first_1 = nn.Conv2d(3 * 2, nf//8, 3, 1, 1, bias=True)
        self.conv_first_2 = nn.Conv2d(nf//8, nf//2, 3, 2, 1, bias=True)
        self.conv_first_3 = nn.Conv2d(nf//2, nf, 3, 2, 1, bias=True)
        self.fftblock1 = FFCResnetBlock(nf)
        self.multiblock1 = Multi_Conv_Block(nf)

        self.fftblock2 = FFCResnetBlock(nf)
        self.multiblock2 = Multi_Conv_Block(nf)

        self.fftblock3 = FFCResnetBlock(nf)
        self.multiblock3 = Multi_Conv_Block(nf)

        self.fftblock4 = FFCResnetBlock(nf)
        self.multiblock4 = Multi_Conv_Block(nf)

        self.fftblock5 = FFCResnetBlock(nf)
        self.multiblock5 = Multi_Conv_Block(nf)

        self.fftblock9 = FFCResnetBlock(nf)
        self.multiblock9 = Multi_Conv_Block(nf)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.Stage2MOE = Stage2MOE(nf)
        self.FrGuide = Stage2Fr(nf)

    def forward(self, x, fr_refine):
        x1 = self.lrelu(self.conv_first_1(x))
        x2 = self.lrelu(self.conv_first_2(x1))
        x3 = self.lrelu(self.conv_first_3(x2))
        f1 = self.fftblock1(x3)
        f1 = self.FrGuide(f1, fr_refine)
        m1 = self.multiblock1(x3)
        f2 = self.fftblock2(f1)
        f2 = self.FrGuide(f2, fr_refine)
        m2 = self.multiblock2(m1)
        f3 = self.fftblock3(f2)
        f3 = self.FrGuide(f3, fr_refine)
        m3 = self.multiblock3(m2)
        f4 = self.fftblock4(f3)
        f4 = self.FrGuide(f4, fr_refine)
        m4 = self.multiblock4(m3)
        f5 = self.fftblock5(f4)
        f5 = self.FrGuide(f5, fr_refine)
        m5 = self.multiblock5(m4)
        f6 = self.fftblock9(f5)
        f6 = self.FrGuide(f6, fr_refine)
        m6 = self.multiblock9(m5)
        m_final = self.Stage2MOE([m1, m2, m3, m4, m5], m6)
        return x1, x2,x3,f6,m_final


