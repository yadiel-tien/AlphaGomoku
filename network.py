import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        """特征提取，输入[B,C,H,W],输出[B,n_filters,H,W]"""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 如果输入是[C,H,W],则补上B=1，->[1,C,H,W]
        if x.ndimension() == 3:
            x = x.unsqueeze(0)
        return self.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, n_filters):
        """残差块处理特征，输入输出结构一样[B,n_filters,H,W]"""
        super().__init__()
        self.conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y += x  # 残差连接
        return self.relu2(y)


class PolicyHead(nn.Module):
    def __init__(self, in_channels, n_filters, n_cells):
        """
        策略头，输出动作策略概率分布。[B,in_channels,H,W]->[B,H*W]/[H*W]
        :param in_channels: 输入通道数
        :param n_filters: 中间特征通道数量
        :param n_cells: 动作空间大小,应为H*W
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, n_filters, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(n_cells * n_filters, n_cells)

    def forward(self, x):
        # x:[B,in_channels,H,W]
        x = self.relu(self.bn(self.conv(x)))
        # x:[B,n_filters,H,W]
        x = x.reshape(x.shape[0], -1)  # 展平
        # x:[B,n_filters*H*W]
        x = self.fc(x)
        # x:[B,H*W]
        probs = torch.softmax(x, dim=-1)  # 在最后一维上应用 softmax
        # probs:[B,H*W]
        probs = probs.squeeze(0)  # 如果批量为1，就消除批量层
        # 如果B=1，则[1,H*W]->[H*W]
        return probs


class ValueHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_cells):
        """价值头,输出当前盘面价值。【B,in_channels,H,W]->[B]"""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(1)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(n_cells, hidden_channels)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_channels, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x:[B,in_channels,H,W]
        x = self.relu1(self.bn(self.conv(x)))
        # x:[B,1,H,W]
        x = x.reshape(x.shape[0], -1)
        # x:[B,H*W]
        x = self.fc2(self.relu2(self.fc1(x)))
        # x:[B,1]
        value = self.tanh(x).reshape(-1)
        # x:[B]
        return value


class Net(nn.Module):
    def __init__(self, n_filters, n_cells=15 * 15, n_res_blocks=7):
        """
        类alpha zero结构，双头输出policy和value
        :param n_filters: 卷积层通道数
        :param n_cells: 动作空间大小，应等于H*W
        :param n_res_blocks: 残差块个数
        """
        super().__init__()
        self.conv_block = ConvBlock(2, n_filters)
        self.res_blocks = nn.ModuleList([ResBlock(n_filters) for _ in range(n_res_blocks)])
        self.policy = PolicyHead(n_filters, 32, n_cells)
        self.value = ValueHead(n_filters, 256, n_cells)

    def forward(self, x):
        x = self.conv_block(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        return self.policy(x), self.value(x)
