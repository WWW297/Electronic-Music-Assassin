import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class DilatedResConv(nn.Module):
    def __init__(self, channels, dilation=1, padding=1, kernel_size=3):
        super().__init__()
        in_channels = channels
        self.dilated_conv = nn.Conv1d(in_channels, channels, kernel_size=kernel_size, stride=1,padding=dilation * padding, dilation=dilation, bias=True)
        self.conv_1x1 = nn.Conv1d(in_channels, channels,kernel_size=1, bias=True)

    def forward(self, input):
        x = self.dilated_conv(input)
        x = F.relu(x)
        x = self.conv_1x1(x)

        return input + x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_blocks = 2
        self.n_layers = 4
        self.channels = 128
        self.latent_channels =64

        layers = []
        for block in range(self.n_blocks):
            for i in range(self.n_layers):
                layers.append(DilatedResConv(self.channels))
        self.dilated_convs = nn.Sequential(*layers)

        conv2d_layers=[]
        conv2d_layers.append(nn.Conv2d(1, 8, kernel_size=3, stride=1,padding=1))
        conv2d_layers.append(nn.ReLU())
        conv2d_layers.append(nn.Conv2d(8, 16, kernel_size=3, stride=1,padding=1))
        conv2d_layers.append(nn.ReLU())
        conv2d_layers.append(nn.Conv2d(16, 1, kernel_size=3, stride=1,padding=1))
        conv2d_layers.append(nn.ReLU())
        self.conv2d_layers = nn.Sequential(*conv2d_layers)
        
        self.start = nn.Conv1d(257, self.channels, kernel_size=3, stride=1,padding=1)
        self.conv_1x1 = nn.Conv1d(self.channels, self.latent_channels, 1)
        self.pool = nn.AvgPool1d(4)

    def forward(self, x):
        x = self.start(x)
        x = self.dilated_convs(x)
        x = x.view(x.size(0),1,x.size(1),x.size(2))
        x = self.conv2d_layers(x)
        x = x.view(x.size(0),x.size(2),x.size(3))
        x = self.conv_1x1(x)
        x = self.pool(x)
        x = F.tanh(x)
        return x


class ZDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        convs = []
        for i in range(5):
            in_channels = 257 if i == 0 else 128
            convs.append(nn.Conv1d(in_channels, 128, 1))
            convs.append(nn.ELU())
        convs.append(nn.Conv1d(128, 2, 1))

        self.convs = nn.Sequential(*convs)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, z):
        z = self.dropout(z)
        logits = self.convs(z)  # (N, n_classes, L)
        mean = logits.mean(2)
        return mean
        


class WavenetLayer(nn.Module):
    def __init__(self, residual_channels, skip_channels, cond_channels,
                 kernel_size=2, dilation=1):
        super(WavenetLayer, self).__init__()
        self.causal = nn.Conv1d(residual_channels, residual_channels,
                                   kernel_size, padding=1,dilation=dilation, bias=True)
        self.condition = nn.Conv1d(cond_channels, residual_channels,
                                   kernel_size=1, bias=True)
        self.residual = nn.Conv1d(residual_channels, residual_channels,
                                  kernel_size=1, bias=True)
        self.skip = nn.Conv1d(residual_channels, skip_channels,
                              kernel_size=1, bias=True)

    def _condition(self, x, c, f):
        c = f(c)
        x = torch.cat((x,c),1)
        return x

    def forward(self, x, c=None):
        x = self.causal(x)
        x = self._condition(x, c, self.condition)
        gate, output = x.chunk(2, 1)

        gate = torch.sigmoid(gate)
        output = torch.tanh(output)
        x = gate+output

        residual = self.residual(x)
        skip = self.skip(x)

        return residual, skip


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks = 4
        self.layer_num = 3
        self.kernel_size = 3
        self.skip_channels = 128
        self.residual_channels = 128
        self.cond_channels = 64

        layers = []
        for _ in range(self.blocks):
            for i in range(self.layer_num):
                layers.append(WavenetLayer(self.residual_channels, self.skip_channels, self.cond_channels,
                                            self.kernel_size))
        self.layers = nn.ModuleList(layers)

        self.first_conv = nn.Conv1d(257, self.residual_channels, kernel_size=self.kernel_size,padding=1)
        self.skip_conv = nn.Conv1d(self.residual_channels, self.skip_channels, kernel_size=1)
        self.condition = nn.Conv1d(self.cond_channels, self.skip_channels, kernel_size=1)
        self.fc = nn.Conv1d(self.skip_channels, self.skip_channels, kernel_size=1)
        self.logits = nn.Conv1d(self.skip_channels, 257, kernel_size=1)
        self.conv1d1 = nn.Conv1d(257, 257, kernel_size=3,padding=1)
        self.conv1d2 = nn.Conv1d(257, 257, kernel_size=3,padding=1)

    def _condition(self, x, c, f):
        c = f(c)
        x = x + c
        return x

    @staticmethod
    def _upsample_cond(x, c):
        bsz, channels, length = x.size()
        cond_bsz, cond_channels, cond_length = c.size()
        assert bsz == cond_bsz
        if cond_length != 1:
            c = c.unsqueeze(3).repeat(1, 1, 1, length // cond_length)
            c = c.view(bsz, cond_channels, length)
        return c


    def forward(self, x, c=None):
        c = self._upsample_cond(x, c)

        residual = self.first_conv(x)
        skip = self.skip_conv(residual)

        for layer in self.layers:
            r, s = layer(residual, c)
            residual = residual + r
            skip = skip + s

        skip = F.relu(skip)
        skip = self.fc(skip)
        skip = self._condition(skip, c, self.condition)
        skip = F.relu(skip)
        skip = self.logits(skip)
        skip = self.conv1d1(skip)
        skip = F.relu(skip)
        skip = self.conv1d2(skip)
        skip = F.tanh(skip)
        return skip
