import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeGuidedIRRefinement(nn.Module):
    def __init__(self, channels=3):
        super(EdgeGuidedIRRefinement, self).__init__()

        # **边缘引导特征提取**
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # **注意力机制 (CBAM)**
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 32, 1),
            nn.Sigmoid()
        )

        self.spatial_att = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.direction_convs = nn.ModuleList([
            nn.Conv2d(channels, 8, kernel_size=(1, 5), padding=(0,2)),  # 水平向
            nn.Conv2d(channels, 8, kernel_size=(5, 1), padding=(2,0)),  # 垂直向
            nn.Conv2d(channels, 8, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(channels, 8, kernel_size=(3, 1), padding=(1, 0))
        ])

        # **融合 & 输出**
        self.refine_conv = nn.Conv2d(32, channels, kernel_size=1)
        self.refine_conv_ir = nn.Conv2d(32, channels, kernel_size=1)

    def forward(self, ir, edge):
        """
        :param ir: 原始红外图像 (b, 3, h, w)
        :param edge: 由Output1提取的边缘图 (b, 3, h, w)
        :return: 细化后的红外图 (b, 3, h, w)
        """

        # 边缘引导特征提取
        edge_feat = self.edge_conv(edge)
        # 通道注意力
        att_channel = self.channel_att(edge_feat)
        edge_feat = edge_feat * att_channel
        ir_feat = [conv(ir) for conv in self.direction_convs]
        ir_feat = torch.cat(ir_feat, dim=1)
        # 结合红外图信息
        refined_ir = self.refine_conv(edge_feat) + self.refine_conv_ir(ir_feat) + edge # 残差连接，防止信息丢失

        return refined_ir

if __name__ == '__main__':
    net = EdgeGuidedIRRefinement()
    ir = torch.rand(4, 3, 256, 256)
    edge = torch.rand(4, 3, 256, 256)
    out = net(ir, edge)
    print(out)