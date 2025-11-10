# ========================计算params与FLOPs===========================
import torch
from torchvision.models import resnet18
from thop import profile
from models.archs.APMoENet import APMoENet
model = APMoENet(nf=64)
input = torch.randn(1, 3, 256, 256)
input2 = torch.randn(1, 3, 256, 256)
flops, params = profile(model, inputs=(input, input2))
print('flops:{}'.format(flops))
print('flops (G):{}'.format(flops / 1e9))
print('params:{}'.format(params))
