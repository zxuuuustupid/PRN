import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        # 收缩子模块
        self.shrinkage = Shrinkage(out_channels, gap_size=1)
        # 卷积、卷积、收缩
        self.residual_function = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            self.shrinkage
        )
        # 连接点
        self.shortcut = nn.Sequential()

        # 通过1*1卷积来统一维度
        if stride != 1 :
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

# 收缩子模块
class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(gap_size) # 实现GAP
        self.fc = nn.Sequential(  # 学习阈值的两层FC
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x  # 原始值 x
        x = torch.abs(x)  # 取绝对值
        x_abs = x
        x = self.gap(x)  # 全局平均池化
        x = torch.flatten(x, 1)  # 展开
        # average = torch.mean(x, dim=1, keepdim=True)
        average = x
        x = self.fc(x)  # 通过两层FC
        x = torch.mul(average, x)  # 与绝对值点乘
        x = x.unsqueeze(2)  # 增加两个维度，获得阈值向量
        # 软阈值化
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)  # 缩放系数
        x = torch.mul(torch.sign(x_raw), n_sub)  # 对原始值x进行缩放
        return x
class RSNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 4

        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 4, kernel_size=3, padding=1, stride=2,bias=False))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 4, 2)
        self.conv3_x = self._make_layer(block, 8, 2)
        self.conv4_x = self._make_layer(block, 16, 2)
        self.bn = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
       # self.fc = nn.Linear(64 * block.expansion, 8)
        self.fc2 = nn.Linear(16, 1)

    def _make_layer(self, block, out_channels,  stride):
        """make rsnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual shrinkage block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a rsnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        layers = block(self.in_channels, out_channels, stride)
        self.in_channels = out_channels


        return nn.Sequential(layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output=self.bn(output)
        output = self.relu(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        #output = self.fc(output)
        output = self.fc2(output)
        output = F.sigmoid(output)
        return output

def rsnet():
    """ return a RSNet 18 object
    """
    return RSNet(BasicBlock, [2, 2, 2, 2])




#关系模块


    
    
