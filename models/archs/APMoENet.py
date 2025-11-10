import functools
import torch.nn
import models.archs.arch_util as arch_util
from models.archs.StateTwoBlock import *
import torch.nn.functional as F
from models.archs.FuseModel import *
from models.archs.Sobel import *
from models.archs.FrRefineBlock import *
from models.archs.StageOne import *

class APMoENet(nn.Module):
    def __init__(self, nf=64):
        super(APMoENet, self).__init__()
        self.StageOne = StageOneNet(8)
        self.sigmoid = torch.nn.Sigmoid()
        self.nf = nf
        self.ffcmodule = DualGenerator()
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, 1)
        self.upconv1 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=True)
        self.upconv1d = nn.Conv2d(nf, nf // 2, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=True)
        self.upconv2d = nn.Conv2d(nf//2, nf // 8, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf // 4, nf // 8, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_last = nn.Conv2d(nf // 8, 3, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.FuseMoE = FuseMoE(in_channels=64, num_experts=4, k=3)
        self.alpha = nn.Parameter(torch.ones(1)) 
        self.beta = nn.Parameter(torch.ones(1))
        self.Sobel = LearnableSobel()
        self.FrReineBlock = EdgeGuidedIRRefinement()

    def get_amplitude(self, x, fr):
        _, _, H, W = x.shape
        # 得到一个映射图map
        Stage1_map = self.StageOne(x, fr)
        Stage1_map = self.sigmoid(Stage1_map)
        img_s1 = x / (Stage1_map + 0.00000001)
        x_stage1 = img_s1
        rate = 2 ** 3
        pad_h = (rate - H % rate) % rate
        pad_w = (rate - W % rate) % rate
        if pad_h != 0 or pad_w != 0:
            x_stage1 = F.pad(x_stage1, (0, pad_w, 0, pad_h), "reflect")
            x = F.pad(x, (0, pad_w, 0, pad_h), "reflect")
        # 阶段一输出 残差连接
        return x_stage1, x

    def forward(self, x, fr):

        #  ************************  Stage one ************************
        _, _, H, W = x.shape
        x_stage1, x = self.get_amplitude(x, fr)
        #  ************************  Middle Module ************************
        output1_edge = self.Sobel(x_stage1)  # 阶段一输出的边缘图
        fr_refine = self.FrReineBlock(fr, output1_edge)
        fr_refine_ori = fr_refine
        #  ************************  Stage two ************************
        x1,x2,x3,f,s = self.ffcmodule(torch.cat((x_stage1,x),dim=1), fr_refine)
        fr_resize = F.interpolate(fr_refine, size=(f.shape[2], f.shape[3]), mode='bilinear', align_corners=False)
        F_att = self.FuseMoE(f, s, fr_resize)
        f_weighted = f * F_att[:, 0:1, :, :]
        s_weighted = s * F_att[:, 1:2, :, :]
        # 最终融合
        fea= f_weighted * self.alpha + s_weighted * self.beta # (b, 64, h, w)
        out_noise = self.recon_trunk(fea)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(torch.cat((out_noise, x3), dim=1))))
        out_noise = self.lrelu(self.upconv1d(out_noise))
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(torch.cat((out_noise, x2), dim=1))))
        out_noise = self.lrelu(self.upconv2d(out_noise))
        out_noise = self.lrelu(self.HRconv(torch.cat((out_noise, x1), dim=1)))
        out_noise = self.lrelu(self.conv_last(out_noise))
        out_noise = out_noise + x
        out_noise = out_noise[:, :, :H, :W]

        return out_noise, x_stage1, fr, fr_refine_ori, output1_edge
