from timm.models.layers import trunc_normal_
import math
from spatial_attention import *

class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
    def __init__(self, dim_in, dim_out, x=8, y=8):
        super().__init__()
        c_dim_in = dim_in // 4
        k_size = 3
        pad = (k_size - 1) // 2
        self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y), requires_grad=True)
        nn.init.ones_(self.params_xy)
        self.conv_xy = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, k_size, 1, pad, groups=c_dim_in),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, 1, 1, 0)
        )

        self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x), requires_grad=True)
        nn.init.ones_(self.params_zx)
        self.conv_zx = nn.Sequential(
            nn.Conv1d(c_dim_in, c_dim_in, k_size, 1, pad, groups=c_dim_in),
            nn.GELU(),
            nn.Conv1d(c_dim_in, c_dim_in, 1, 1, 0)
        )

        self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y), requires_grad=True)
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(
            nn.Conv1d(c_dim_in, c_dim_in, k_size, 1, pad, groups=c_dim_in),
            nn.GELU(),
            nn.Conv1d(c_dim_in, c_dim_in, 1, 1, 0)
        )

        self.dw = nn.Sequential(
                nn.Conv2d(c_dim_in, c_dim_in, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(c_dim_in, c_dim_in, 3, 1, 1, groups=c_dim_in)
        )
        
        self.norm1 = LayerNorm(dim_in)
        self.norm2 = LayerNorm(dim_in)
        self.ldw = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, 3, 1, 1, groups=dim_in),
                nn.GELU(),
                nn.Conv2d(dim_in, dim_out, 1, 1, 0),
        )
        
    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        # ----------xy---------- #
        params_xy = self.params_xy
        x1 = x1 * self.conv_xy(F.interpolate(params_xy, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        # ----------zx---------- #
        x2 = x2.permute(0, 3, 1, 2)
        params_zx = self.params_zx
        x2 = x2 * self.conv_zx(F.interpolate(params_zx, size=x2.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        # ----------zy---------- #
        x3 = x3.permute(0, 2, 1, 3)
        params_zy = self.params_zy
        x3 = x3 * self.conv_zy(F.interpolate(params_zy, size=x3.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        # ----------dw---------- #
        x4 = self.dw(x4)
        # --------concat-------- #
        x = torch.cat([x1, x2, x3, x4], dim=1)
        # ---------ldw---------- #
        x = self.norm2(x)
        x = self.ldw(x)
        return x

class GHPA_InvRes(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()
        hidden_dim = round(inp * 6)
        self.use_res_connect = inp == oup
        self.conv = nn.Sequential(
            #-----------------------------------#
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            LayerNorm(hidden_dim),
            nn.ReLU6(inplace=True),
            #--------------------------------------------#
            Grouped_multi_axis_Hadamard_Product_Attention(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim),
            nn.ReLU6(inplace=True),
            #-----------------------------------#
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            LayerNorm(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class SE_Block(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=channel, out_features=channel // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel // 2, out_features=channel),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        residual = x
        max_out = self.mlp(self.max_pool(x).reshape(b, -1))
        avg_out = self.mlp(self.avg_pool(x).reshape(b, -1))
        channel_out = self.sigmoid(max_out + avg_out).reshape(b, c, 1, 1)
        x = channel_out * x
        x = x + residual
        return x

class GALUNet(nn.Module):
    def __init__(self, num_classes=8, input_channels=1, c_list=[8, 24, 40, 64, 88, 112]):
        super().__init__()
        self.encoder1 = nn.Conv2d(input_channels, c_list[0], 3, 1, 1)
        self.encoder2 = GHPA_InvRes(c_list[0], c_list[1])
        self.encoder3 = GHPA_InvRes(c_list[1], c_list[2])
        self.encoder4 = GHPA_InvRes(c_list[2], c_list[3])
        self.encoder5 = GHPA_InvRes(c_list[3], c_list[4])
        self.encoder6 = GHPA_InvRes(c_list[4], c_list[5])

        self.decoder1 = Grouped_multi_axis_Hadamard_Product_Attention(c_list[5], c_list[4])
        self.decoder2 = Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[3])
        self.decoder3 = Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[2])
        self.decoder4 = nn.Conv2d(c_list[2], c_list[1], 3, 1, 1)
        self.decoder5 = nn.Conv2d(c_list[1], c_list[0], 3, 1, 1)
        self.decoder6 = nn.Conv2d(c_list[0], num_classes, 1, 1, 0)

        self.CA1 = SE_Block(c_list[5])
        self.CA2 = SE_Block(c_list[4])
        self.CA3 = SE_Block(c_list[3])
        self.CA4 = SE_Block(c_list[2])
        self.CA5 = SE_Block(c_list[1])
        self.CA6 = SE_Block(c_list[0])

        self.SA1 = NONLocalBlock(c_list[5])
        self.SA2 = PPAG(c_list[4])
        self.SA3 = PPAG(c_list[3])
        self.SA4 = PPAG(c_list[2])
        self.SA5 = PPAG(c_list[1])
        self.SA6 = PPAG(c_list[0])

        self.gt_conv1 = nn.Conv2d(c_list[4], num_classes, 1, 1, 0)
        self.gt_conv2 = nn.Conv2d(c_list[3], num_classes, 1, 1, 0)
        self.gt_conv3 = nn.Conv2d(c_list[2], num_classes, 1, 1, 0)
        self.gt_conv4 = nn.Conv2d(c_list[1], num_classes, 1, 1, 0)
        self.gt_conv5 = nn.Conv2d(c_list[0], num_classes, 1, 1, 0)

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2
        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4
        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, c4, H/32, W/32
        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32

        out = self.SA1(out)

        out5 = F.gelu(self.dbn1(self.decoder1(self.CA1(out))))  # b, c4, H/32, W/32
        t5 = self.SA2(t5, out5)
        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32
        gt_pre5 = self.gt_conv1(out5)
        gt_pre5 = F.interpolate(gt_pre5, scale_factor=32, mode='bilinear', align_corners=True)

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(self.CA2(out5))), scale_factor=(2, 2), mode='bilinear', align_corners=True))  # b, c3, H/16, W/16
        t4 = self.SA3(t4, out4)
        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16
        gt_pre4 = self.gt_conv2(out4)
        gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode='bilinear', align_corners=True)

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(self.CA3(out4))), scale_factor=(2, 2), mode='bilinear', align_corners=True))  # b, c2, H/8, W/8
        t3 = self.SA4(t3, out3)
        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8
        gt_pre3 = self.gt_conv3(out3)
        gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode='bilinear', align_corners=True)

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(self.CA4(out3))), scale_factor=(2, 2), mode='bilinear', align_corners=True))  # b, c1, H/4, W/4
        t2 = self.SA5(t2, out2)
        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4
        gt_pre2 = self.gt_conv4(out2)
        gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode='bilinear', align_corners=True)

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(self.CA5(out2))), scale_factor=(2, 2), mode='bilinear', align_corners=True))  # b, c0, H/2, W/2
        t1 = self.SA6(t1, out1)
        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2
        gt_pre1 = self.gt_conv5(out1)
        gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode='bilinear', align_corners=True)

        out0 = F.interpolate(self.decoder6(self.CA6(out1)), scale_factor=(2, 2), mode='bilinear', align_corners=True)  # b, num_class, H, W

        return (gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1), out0
