import torch
import torch.nn as nn


def get_edge(img, eps=1e-6):
    """ 计算 Sobel 边缘图（先转灰度，再计算梯度） """
    if img.shape[1] == 3:  # 代码输出是3通道的
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32, device=img.device).view(1, 3, 1, 1)
        gray_img = torch.sum(img * weights, dim=1, keepdim=True)  # (b, 1, h, w)
    elif img.shape[1] == 1:  # 单通道
        gray_img = img
    else:
        raise ValueError("输入通道数必须为 1 或 3")

    # Sobel 计算
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=img.device).view(1, 1, 3,
                                                                                                              3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=img.device).view(1, 1, 3,
                                                                                                              3)
    grad_x = F.conv2d(gray_img, sobel_x, padding=1)
    grad_y = F.conv2d(gray_img, sobel_y, padding=1)
    grad_x = grad_x / (torch.abs(grad_x).max() + eps)
    grad_y = grad_y / (torch.abs(grad_y).max() + eps)

    return grad_x, grad_y  # 返回单通道边缘图 (b, 1, h, w)


# **改进的边缘一致性损失**
def edge_consistency_loss(gt, output):
    """
    计算边缘一致性损失，包括：
    1. **平滑 L1 损失（Huber Loss）**：防止梯度过大导致不稳定
    2. **归一化梯度差异**：限制梯度范围，防止 NaN
    """
    gt_grad_x, gt_grad_y = get_edge(gt)
    out_grad_x, out_grad_y = get_edge(output)
    # **平滑 L1 损失**
    loss_x = F.smooth_l1_loss(out_grad_x, gt_grad_x, beta=1.0)  # beta 控制 L1/L2 切换点
    loss_y = F.smooth_l1_loss(out_grad_y, gt_grad_y, beta=1.0)
    # **组合损失**
    total_loss = loss_x + loss_y

    return total_loss


import kornia.color as kcolor


def lab_loss(fake_rgb, real_rgb):

    # 1. RGB → Lab 转换（输入需归一化到 [0, 1]）
    fake_lab = kcolor.rgb_to_lab(fake_rgb)
    real_lab = kcolor.rgb_to_lab(real_rgb)
    # 2. 拆分通道
    fake_L, fake_a, fake_b = fake_lab[:, :1], fake_lab[:, 1:2], fake_lab[:, 2:3]
    real_L, real_a, real_b = real_lab[:, :1], real_lab[:, 1:2], real_lab[:, 2:3]
    # 3. 计算每通道损失
    loss_L = F.l1_loss(fake_L, real_L)
    loss_ab = F.l1_loss(fake_a, real_a) + F.l1_loss(fake_b, real_b)
    # 4. 总损失加权组合
    total_lab_loss = (loss_L + loss_ab) * 0.01

    return total_lab_loss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


##############
class CharbonnierLoss2(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss2, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


import torchvision
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        # self.criterion = nn.L1Loss()
        self.criterion = nn.L1Loss(reduction='sum')
        self.criterion2 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward2(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion2(x_vgg[i], y_vgg[i].detach())
        return loss


import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram

class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, inpput, gt):
        style_loss = self.l1(gram_matrix(inpput),
                             gram_matrix(gt))
        return style_loss