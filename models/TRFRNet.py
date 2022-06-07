import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import functools
import numpy as np


class TRFRNet(nn.Module):
    def __init__(self, num_classes):
        super(TRFRNet, self).__init__()
        self.encoder = Encoder(num_classes)
        # polyp relevant
        self.TRFR = TRFR()
        # Decoder
        self.decoder = Decoder(num_classes)                                         

    def forward(self, x):
        # x 224
        n5, r5, r4, r3, r2, r1 = self.encoder(x)
        plus = self.TRFR(r1, r2, r3, r4, r5)        
        out1 = self.decoder(n5+plus)
        return out1

    def entropy(self,x):
        n5, r5, r4, r3, r2, r1 = self.encoder(x)
        plus = self.TRFR(r1, r2, r3, r4, r5)
        out1 = self.decoder(n5+plus)
        out = self.decoder(n5)
        return out1, out

class Encoder(nn.Module):
    def __init__(self, num_classes):
        super(Encoder, self).__init__()

        resnet = models.resnet34(pretrained=True)
        
        # Encoder
        self.encoder1_conv = resnet.conv1
        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        # SN
        self.norm1 = DIFD(chan_num=64)
        self.norm2 = DIFD(chan_num=64)
        self.norm3 = DIFD(chan_num=128)
        self.norm4 = DIFD(chan_num=256)
        self.norm5 = DIFD(chan_num=512)

    def forward(self, x):
        e1 = self.encoder1_conv(x)  # 128
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1_pool = self.maxpool(e1)  # 56
        n1 = self.norm1(e1_pool)
        e2 = self.encoder2(n1)
        n2 = self.norm2(e2)
        e3 = self.encoder3(n2)  # 28
        n3 = self.norm3(e3)
        e4 = self.encoder4(n3)  # 14
        n4 = self.norm4(e4)
        e5 = self.encoder5(n4)  # 7
        n5 = self.norm5(e5)

        # return n5, e5-n5, e4-n4, e3-n3, e2-n2, e1_pool-n1, n4,n3,n2,n1
        return n5, e5-n5, e4-n4, e3-n3, e2-n2, e1_pool-n1

class DIFD(nn.Module):
    """DIFD module"""
    def __init__(self, chan_num, is_two=True):
        super(DIFD, self).__init__()

        self.g_fc = nn.Conv1d(chan_num, chan_num, kernel_size=2,
                              bias=False, groups=chan_num)
        self.g_bn = nn.BatchNorm1d(chan_num)

        if is_two is True:
            self.f_fc = nn.Conv1d(chan_num, chan_num, kernel_size=2,
                                  bias=False, groups=chan_num)
            self.f_bn = nn.BatchNorm1d(chan_num)
        else:
            self.f_fc = None

    def forward(self, x):
        b, c, _, _ = x.size()

        """extract feature map statistics"""
        eps=1e-12 # eps is a small value added to the variance to avoid divide-by-zero.
        var = x.contiguous().view(b, c, -1).var(dim=2) + eps
        std = var.sqrt().view(b, c, 1, 1)
        mean = x.contiguous().view(b, c, -1).mean(dim=2).view(b, c, 1, 1)        

        statistics = torch.cat((mean.squeeze(3), std.squeeze(3)), -1)

        g_y = self.g_fc(statistics)
        g_y = self.g_bn(g_y)
        g_y = torch.sigmoid(g_y)
        g_y = g_y.view(b, c, 1, 1)

        if self.f_fc is not None:
            f_y = self.f_fc(statistics)
            f_y = self.f_bn(f_y)
            f_y = torch.sigmoid(f_y)
            f_y = f_y.view(b, c, 1, 1)

            return x * g_y.expand_as(x) + mean.expand_as(x) * (f_y.expand_as(x)-g_y.expand_as(x))
        else:
            return x * g_y.expand_as(x)

class TRFR(nn.Module):
    def __init__(self):
        super(TRFR, self).__init__()
        self.alpha = TRFR_Ext()
        self.fplus = TRFR_Agg()                                         

    def forward(self, r1, r2, r3, r4, r5):
        a1, a2, a3, a4, a5 = self.alpha(r1, r2, r3, r4, r5)
        x1 = r1 *a1.expand_as(r1)
        x2 = r2 *a2.expand_as(r2)
        x3 = r3 *a3.expand_as(r3)
        x4 = r4 *a4.expand_as(r4)
        x5 = r5 *a5.expand_as(r5)
        plus = self.fplus(x1, x2, x3, x4, x5)
        return plus

'''TRFR_Ext
Block to give channel-wise attentions'''
class TRFR_Ext(nn.Module):
    def __init__(self):
        super(TRFR_Ext, self).__init__()
        in_channel = [64, 64, 128, 256, 512]
        self.a1 = PolypIrr(in_channel[0])
        self.a2 = PolypIrr(in_channel[1])
        self.a3 = PolypIrr(in_channel[2])
        self.a4 = PolypIrr(in_channel[3])
        self.a5 = PolypIrr(in_channel[4])

    def forward(self, r1, r2, r3, r4, r5):
        y1 = self.a1(r1)
        y2 = self.a2(r2)
        y3 = self.a3(r3)
        y4 = self.a4(r4)
        y5 = self.a5(r5)
        return y1, y2, y3, y4, y5

'''PolypIrr
SE-like struction to output alpha for each channel indicating polyprelated information'''
class PolypIrr(nn.Module):
    def __init__(self, channel, reduction = 16):
        super(PolypIrr, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, residual):
        b,c,_,_ = residual.size()
        y = self.avg_pool(residual).view(b,c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

'''TRFR_Agg

Integrated Task-relevant Features'''
class TRFR_Agg(nn.Module):
    def __init__(self):
        super(TRFR_Agg, self).__init__()
        ps = 7
        in_channel = [64, 64, 128, 256, 512]
        out_channel = 512
        self.c1 = nn.Sequential(nn.AdaptiveAvgPool2d(ps),
                                nn.Conv2d(in_channel[0],in_channel[0],3,1,1),
                                nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(nn.AdaptiveAvgPool2d(ps),
                                nn.Conv2d(in_channel[1],in_channel[1],3,1,1),
                                nn.ReLU(inplace=True))
        self.c3 = nn.Sequential(nn.AdaptiveAvgPool2d(ps),
                                nn.Conv2d(in_channel[2],in_channel[2],3,1,1),
                                nn.ReLU(inplace=True))
        self.c4 = nn.Sequential(nn.AdaptiveAvgPool2d(ps),
                                nn.Conv2d(in_channel[3],in_channel[3],3,1,1),
                                nn.ReLU(inplace=True))
        self.c5 = nn.Sequential(nn.Conv2d(in_channel[4],in_channel[4],3,1,1),
                                nn.ReLU(inplace=True),
                                NonLocalBlock(in_channels=in_channel[4]))

        self.out = outCombine(1024, out_channel)

    def forward(self, x1, x2, x3, x4, x5):
        nsize = x5.size()[2:]
        tr1 = self.c1(x1)
        tr2 = self.c2(x2)
        tr3 = self.c3(x3)
        tr4 = self.c4(x4)
        tr5 = self.c5(x5)
        tr1 = F.interpolate(tr1, nsize,mode='bilinear', align_corners=True)
        tr2 = F.interpolate(tr2, nsize,mode='bilinear', align_corners=True)
        tr3 = F.interpolate(tr3, nsize,mode='bilinear', align_corners=True)
        tr4 = F.interpolate(tr4, nsize,mode='bilinear', align_corners=True)
        combine = self.out(torch.cat([tr1, tr2, tr3, tr4, tr5], dim=1))
        return combine

class outCombine(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(outCombine, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.dropout = nn.Dropout2d(0.1)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x


"""
Non Local Block

https://arxiv.org/abs/1711.07971
"""


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

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

class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()

        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64, out_channels=64)
        
        self.outconv1 = nn.Sequential(ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
                                      nn.Dropout2d(0.1),
                                      nn.Conv2d(32, num_classes, 1))

    def forward(self, x):
        d5 = self.decoder5(x)
        d4 = self.decoder4(d5)
        d3 = self.decoder3(d4)
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2)
        out1 = self.outconv1(d1)
        return out1

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super(DecoderBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = ConvBlock(in_channels // 4, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x