import torch
import torch.nn as nn
import torch.nn.functional as F


# 可学习边缘提取器
class LearnableSobel(nn.Module):
    def __init__(self, in_channels=3):
        super(LearnableSobel, self).__init__()

        # 初始化 Sobel 过滤器（可学习）
        self.sobel_x = nn.Parameter(torch.tensor([[[[-1, 0, 1],
                                                    [-2, 0, 2],
                                                    [-1, 0, 1]]]], dtype=torch.float32))
        self.sobel_y = nn.Parameter(torch.tensor([[[[-1, -2, -1],
                                                    [0, 0, 0],
                                                    [1, 2, 1]]]], dtype=torch.float32))

        # 将滤波器扩展到所有输入通道
        self.sobel_x = nn.Parameter(self.sobel_x.repeat(in_channels, 1, 1, 1), requires_grad=True)
        self.sobel_y = nn.Parameter(self.sobel_y.repeat(in_channels, 1, 1, 1), requires_grad=True)

    def forward(self, x):
        # 对输入图像进行 Sobel 过滤
        edge_x = F.conv2d(x, self.sobel_x, padding=1, groups=x.shape[1])
        edge_y = F.conv2d(x, self.sobel_y, padding=1, groups=x.shape[1])
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        edge = edge / edge.max()

        return edge


# 测试
if __name__ == "__main__":
    input_tensor = torch.randn(1, 3, 1039, 789)  # 模拟 batch 维度的输入
    model = LearnableSobel()
    edge_output = model(input_tensor)
    print(edge_output.shape)  # 输出: (1, 3, 256, 256)


