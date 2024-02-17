import torch
from torch import nn
# from core.ops.sync_bn import convert_bn
# from core.ops.sync_bn import SyncBatchNorm2d as BatchNorm2d

from torch.nn import BatchNorm2d

def meshgrid(height, width):
    x_t = torch.matmul(
        torch.ones(height, 1), torch.linspace(-1.0, 1.0, width).view(1, width))
    y_t = torch.matmul(
        torch.linspace(-1.0, 1.0, height).view(height, 1), torch.ones(1, width))

    grid_x = x_t.view(1, height, width)
    grid_y = y_t.view(1, height, width)
    return grid_x, grid_y


class VoxelFlow(nn.Module):

    def __init__(self, config):
        super(VoxelFlow, self).__init__()
        self.config = config
        self.input_mean = [0.5 * 255, 0.5 * 255, 0.5 * 255]
        self.input_std = [0.5 * 255, 0.5 * 255, 0.5 * 255]
        # self.input_mean=[123.675, 116.28, 103.53],
        # self.input_std=[58.395, 57.12, 57.375],

        self.syn_type = config.syn_type

        bn_param = config.bn_param
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(
            6, 64, kernel_size=5, stride=1, padding=2, bias=False)
        # self.conv1_bn = BatchNorm2d(64, **bn_param)
        self.conv1_bn = BatchNorm2d(64)

        self.conv2 = nn.Conv2d(
            64, 128, kernel_size=5, stride=1, padding=2, bias=False)
        # self.conv2_bn = BatchNorm2d(128, **bn_param)
        self.conv2_bn = BatchNorm2d(128)

        self.conv3 = nn.Conv2d(
            128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv3_bn = BatchNorm2d(256, **bn_param)
        self.conv3_bn = BatchNorm2d(256)

        self.bottleneck = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bottleneck_bn = BatchNorm2d(256, **bn_param)
        self.bottleneck_bn = BatchNorm2d(256)

        self.deconv1 = nn.Conv2d(
            512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        # self.deconv1_bn = BatchNorm2d(256, **bn_param)
        self.deconv1_bn = BatchNorm2d(256)

        self.deconv2 = nn.Conv2d(
            384, 128, kernel_size=5, stride=1, padding=2, bias=False)
        # self.deconv2_bn = BatchNorm2d(128, **bn_param)
        self.deconv2_bn = BatchNorm2d(128)

        self.deconv3 = nn.Conv2d(
            192, 64, kernel_size=5, stride=1, padding=2, bias=False)
        # self.deconv3_bn = BatchNorm2d(64, **bn_param)
        self.deconv3_bn = BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(VoxelFlow, self).train(mode)

        # 我感觉目前的版本无需要关心并行的细节了
        # if mode:
        #     convert_bn(self, self.config.bn_training, self.config.bn_parallel)
        # else:
        #     convert_bn(self, False, False)

    def get_optim_policies(self):
        outs = []
        outs.extend(
            self.get_module_optim_policies(
                self,
                self.config,
                'model',
            ))
        return outs

    def get_module_optim_policies(self, module, config, prefix):
        weight = []
        bias = []
        bn = []

        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                ps = list(m.parameters())
                weight.append(ps[0])
                if len(ps) == 2:
                    bias.append(ps[1])
            elif isinstance(m, nn.BatchNorm2d):
                bn.extend(list(m.parameters()))

        return [
            {
                'params': weight,
                'lr_mult': config.mult_conv_w[0],
                'decay_mult': config.mult_conv_w[1],
                'name': prefix + " weight"
            },
            {
                'params': bias,
                'lr_mult': config.mult_conv_b[0],
                'decay_mult': config.mult_conv_b[1],
                'name': prefix + " bias"
            },
            {
                'params': bn,
                'lr_mult': config.mult_bn[0],
                'decay_mult': config.mult_bn[1],
                'name': prefix + " bn scale/shift"
            },
        ]

    def forward(self, x, syn_type='inter'):
        input = x
        input_size = tuple(x.size()[2:4])

        # out =(input +2*padding -kernel)/stride + 1 
        # (256+2*2-5)/1 + 1 = 256
        x = self.conv1(x) # (b, 2*3, 256, 256) -> (b, 64, 256, 256)
        x = self.conv1_bn(x)
        conv1 = self.relu(x)
        # (256 + 0 - 2)/2 +1 = 128
        x = self.pool(conv1) # (b, 64, 256, 256) -> (b, 64, 128, 128)

        # (128+2*2-5)/1+1=128
        x = self.conv2(x) # (b, 64,...) -> (b, 128,...)
        x = self.conv2_bn(x)
        conv2 = self.relu(x)
        # (128 + 0 - 2)/2 +1 = 64
        x = self.pool(conv2) # (b, 64, 128, 128) -> (b, 128, 64, 64) 

        # (64+1*2-3)/1+1=64
        x = self.conv3(x) # ->(b, 256, 64, 64)
        x = self.conv3_bn(x)
        conv3 = self.relu(x) 

        x = self.pool(conv3) # ->(b, 256, 32, 32)

        x = self.bottleneck(x) # ->(b, 256, 32, 32)
        x = self.bottleneck_bn(x)
        x = self.relu(x)
 
        x = nn.functional.upsample( # ->(b, 256, 64, 64)
            x, scale_factor=2, mode='bilinear', align_corners=False)

        # out=(input - 1)×stride+k-2p
        # (64-1)*1+3-2=64
        x = torch.cat([x, conv3], dim=1) # ->(b, 512, 64, 64)
        x = self.deconv1(x) # ->(b, 256, 64, 64)
        x = self.deconv1_bn(x)
        x = self.relu(x)

        x = nn.functional.upsample( # ->(b, 256, 128, 128)
            x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([x, conv2], dim=1) # ->(b, 384, 128, 128)
        #(128-1)*1+5-2*2=128
        x = self.deconv2(x) # ->(b, 128, 128, 128)
        x = self.deconv2_bn(x)
        x = self.relu(x)

        x = nn.functional.upsample( # ->(b, 128, 256, 256)
            x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([x, conv1], dim=1) # ->(b, 192, 256, 256)
        x = self.deconv3(x) # ->(b, 64, 256, 256)
        x = self.deconv3_bn(x)
        x = self.relu(x)

        x = self.conv4(x) # ->(b, 3, 256, 256)
        x = nn.functional.tanh(x)

        flow = x[:, 0:2, :, :] 
        # 可能是识别和提取视频中的运动区域，或是第1帧+光流预测图所占权重
        mask = x[:, 2:3, :, :]

        # grid_x：表示每个像素在 x 轴上的坐标值，从左到右依次递增。
        # grid_y：表示每个像素在 y 轴上的坐标值，从上到下依次递增。
        grid_x, grid_y = meshgrid(input_size[0], input_size[1])
        with torch.cuda.device(input.get_device()):
            grid_x = torch.autograd.Variable(
                grid_x.repeat([input.size()[0], 1, 1])).cuda()
            grid_y = torch.autograd.Variable(
                grid_y.repeat([input.size()[0], 1, 1])).cuda()

        flow = 0.5 * flow # 正方向默认是减去光流值表示像素移动

        if self.syn_type == 'inter':
            coor_x_1 = grid_x - flow[:, 0, :, :]
            coor_y_1 = grid_y - flow[:, 1, :, :]
            coor_x_2 = grid_x + flow[:, 0, :, :]
            coor_y_2 = grid_y + flow[:, 1, :, :]
        elif self.syn_type == 'extra':
            coor_x_1 = grid_x - flow[:, 0, :, :] * 2
            coor_y_1 = grid_y - flow[:, 1, :, :] * 2
            coor_x_2 = grid_x - flow[:, 0, :, :] 
            coor_y_2 = grid_y - flow[:, 1, :, :] 

        else:
            raise ValueError('Unknown syn_type ' + self.syn_type)

        # 将光流产生的变化应用到第1个图
        output_1 = torch.nn.functional.grid_sample(
            input[:, 0:3, :, :],
            torch.stack([coor_x_1, coor_y_1], dim=3),
            padding_mode='border')

        # 将光流产生的变化应用到第2个图
        output_2 = torch.nn.functional.grid_sample(
            input[:, 3:6, :, :],
            torch.stack([coor_x_2, coor_y_2], dim=3),
            padding_mode='border')

        # 将 mask 中的像素值从原来的 [0, 1] 区间缩放到 [0.5, 1.0] 区间
        # 使其更好地融合背景
        mask = 0.5 * (1.0 + mask)
        mask = mask.repeat([1, 3, 1, 1])

        x = mask * output_1 + (1.0 - mask) * output_2
        return x
