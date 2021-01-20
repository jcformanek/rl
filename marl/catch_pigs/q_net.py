import torch
import torch.nn as nn
import math

class QNet(nn.Module):
    """
    Convolutional layers with MLP readout.
    """
    def __init__(self, in_dims, out_dim, dropout=0.5):
        super(QNet, self).__init__()
        """
        in_dims :: (int, ... , int)
        out_dim :: int
        """
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.dropout = dropout
        self.flattened = -1

        self.batch_norms = nn.ModuleList()

        self.conv_layers = nn.ModuleList()

        in_channels = self.in_dims[0]
        h = self.in_dims[1]
        w = self.in_dims[2]
        out_channels = 5
        kernel_size = 4
        stride = 1
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.conv_layers.append(conv)
        self.batch_norms.append(nn.BatchNorm2d(out_channels))

        in_channels = out_channels
        h = math.floor((h - (kernel_size - 1) - 1) / stride + 1)
        w = math.floor((w - (kernel_size - 1) - 1) / stride + 1)
        out_channels = 10
        kernel_size = 4
        stride = 1
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.conv_layers.append(conv)
        self.batch_norms.append(nn.BatchNorm2d(out_channels))

        h = math.floor((h - (kernel_size - 1) - 1) / stride + 1)
        w = math.floor((w - (kernel_size - 1) - 1) / stride + 1)

        self.mlp = nn.ModuleList()

        self.flattened = h*w*out_channels
        in_feat = self.flattened
        out_feat = 100
        layer = nn.Linear(in_feat, out_feat)
        self.mlp.append(layer)

        in_feat = out_feat
        out_feat = self.out_dim
        layer = nn.Linear(in_feat, out_feat)
        self.mlp.append(layer)


    def forward(self, x):

        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            x = self.batch_norms[i](x)
            x = torch.nn.functional.dropout(x, self.dropout)

        x = x.view(-1, self.flattened)

        for i in range(len(self.mlp) -1):
            x = self.mlp[i](x)
            x = torch.nn.functional.dropout(x, self.dropout)

        x = self.mlp[-1](x)

        return x


        






