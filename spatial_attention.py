from torch import nn
import torch
import torch.nn.functional as F
from torch.nn import init
from utils import LayerNorm

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class NONLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.inter_channels = in_channels // 4

        self.g = nn.Sequential(
            nn.Conv2d(in_channels, self.inter_channels, 1, 1, 0),
            nn.MaxPool2d(kernel_size=2),
        )
        self.phi = nn.Sequential(
            nn.Conv2d(in_channels, self.inter_channels, 1, 1, 0),
            nn.MaxPool2d(kernel_size=2),
        )
        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1, 1, 0)
        self.W = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_channels, 1, 1, 0),
            LayerNorm(in_channels),
        )

        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        for m in self.children():
            m.apply(weights_init_kaiming)

    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

class Attention_Gate(nn.Module):
    def __init__(self, x_channels, g_channels, inter_channels):
        super().__init__()
        self.W = nn.Sequential(
            nn.Conv2d(x_channels, x_channels, 1, 1, 0),
            LayerNorm(x_channels),
        )
        self.theta = nn.Conv2d(x_channels, inter_channels, 1, 1, 0, bias=True)
        self.phi = nn.Conv2d(g_channels, inter_channels, 1, 1, 0, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, 1, 1, 0, bias=True)

        for m in self.children():
            m.apply(weights_init_kaiming)

    def forward(self, x, g):
        input_size = x.size()
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode='bilinear', align_corners=True)
        f = F.relu(theta_x + phi_g, inplace=True)
        sigm_psi_f = F.sigmoid(self.psi(f))
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode='bilinear', align_corners=True)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y

class PPAG(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.gate_block_1 = Attention_Gate(
            x_channels=in_size,
            g_channels=in_size,
            inter_channels=in_size,
        )
        self.gate_block_2 = Attention_Gate(
            x_channels=in_size,
            g_channels=in_size,
            inter_channels=in_size,
        )
        self.combine_gates = nn.Sequential(
            nn.Conv2d(in_size, in_size, 1, 1, 0),
            LayerNorm(in_size),
            nn.ReLU(inplace=True)
        )

        for m in self.children():
            if m.__class__.__name__.find('Attention_Gate') != -1:
                continue
            m.apply(weights_init_kaiming)

    def forward(self, input, gating_signal):
        gate_1 = self.gate_block_1(input, gating_signal)
        gate_2 = self.gate_block_2(input, gating_signal)

        return self.combine_gates(gate_1 + gate_2)
