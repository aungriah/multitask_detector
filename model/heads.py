import torch
import torch.nn as nn
import numpy as np


# ------------------------------------------------------------------------------------------------------------------------ #
# -------------------------------------------------- Lane detector head -------------------------------------------------- #

class AegisLaneHead(nn.Module):
    def __init__(self, config):
        """
        config: hyperparameters of model, including parameters to define lane detection head
        returns lane detection head architecture
        """

        self.num_resnet_layers = config['arch']['backbone_layers']
        self.cls_dim = (config['arch']['griding_num']+1, config['arch']['cls_num_per_lane'], config['arch']['num_lanes'])
        self.total_dim = np.prod(self.cls_dim)
        self.use_aux = config['arch']['use_aux']
        self.fc_dim = 8*(config['arch']['input_width']/32)*(config['arch']['input_height']/32)
        assert self.fc_dim - int(self.fc_dim) == 0, "Watch out for the input dimension!"
        self.fc_dim = int(self.fc_dim)
        super(AegisLaneHead, self).__init__()

        if self.use_aux:
            self.aux_header2 = torch.nn.Sequential(
                conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1) if self.num_resnet_layers in ['34','18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_header3 = torch.nn.Sequential(
                conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1) if self.num_resnet_layers in ['34','18'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_header4 = torch.nn.Sequential(
                conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1) if self.num_resnet_layers in ['34','18'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            self.aux_combine = torch.nn.Sequential(
                conv_bn_relu(384, 256, 3,padding=2,dilation=2),
                conv_bn_relu(256, 128, 3,padding=2,dilation=2),
                conv_bn_relu(128, 128, 3,padding=2,dilation=2),
                conv_bn_relu(128, 128, 3,padding=4,dilation=4),
                torch.nn.Conv2d(128, self.cls_dim[-1] + 1,1)
                # output : n, num_of_lanes+1, h, w
            )
            initialize_weights(self.aux_header2, self.aux_header3, self.aux_header4, self.aux_combine)

        # input : nchw
        # output: (w+1) * sample_rows * 4

        self.cls = torch.nn.Sequential(
            torch.nn.Linear(self.fc_dim, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.total_dim),
        )
        initialize_weights(self.cls)

        self.pool = torch.nn.Conv2d(512,8,1) if self.num_resnet_layers in ['34','18'] else torch.nn.Conv2d(2048,8,1)
        # 1/32,2048 channel
        # 288,800 -> 9,40,2048
        # (w+1) * sample_rows * 4
        # 37 * 10 * 4

    def forward(self, x2, x3, fea):
        # n c h w - > n 2048 sh sw
        # -> n 2048
        if self.use_aux:
            x2 = self.aux_header2(x2)
            x3 = self.aux_header3(x3)
            x3 = torch.nn.functional.interpolate(x3, scale_factor=2, mode='bilinear')
            x4 = self.aux_header4(fea)
            x4 = torch.nn.functional.interpolate(x4, scale_factor=4, mode='bilinear')
            aux_seg = torch.cat([x2, x3, x4], dim=1)
            aux_seg = self.aux_combine(aux_seg)
        else:
            aux_seg = None

        fea = self.pool(fea).view(-1, self.fc_dim)

        group_cls = self.cls(fea).view(-1, *self.cls_dim)

        if self.use_aux:
            return group_cls, aux_seg

        return group_cls

def initialize_weights(*models):
    for model in models:
        real_init_weights(model)

def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unkonwn module', m)

class conv_bn_relu(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False):
        super(conv_bn_relu,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,out_channels, kernel_size,
            stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# ------------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------- Object detector head ------------------------------------------------- #

# TODO define this in config
BN_MOMENTUM = 0.1

class AegisObjHead(nn.Module):
    def __init__(self, config):
        super(AegisObjHead, self).__init__()
        """
        config: hyperparameters of model, including parameters to define object detection head
        returns object detection head architecture
        """


        self.inplanes = config['arch']['inplanes']
        self.deconv_with_bias = False
        self.head_conv = config['arch']['head_conv']
        self.num_classes = config['arch']['num_obj_classes']

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(3, [256, 256, 256], [4, 4, 4])
        # self.final_layer = []

        if self.head_conv > 0:
            # heatmap layers
            self.hmap = nn.Sequential(nn.Conv2d(256, self.head_conv, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.head_conv, self.num_classes, kernel_size=1))
            self.hmap[-1].bias.data.fill_(-2.19)
            # regression layers
            self.regs = nn.Sequential(nn.Conv2d(256, self.head_conv, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.head_conv, 2, kernel_size=1))
            self.w_h_ = nn.Sequential(nn.Conv2d(256, self.head_conv, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.head_conv, 2, kernel_size=1))
        else:
            # heatmap layers
            self.hmap = nn.Conv2d(in_channels=256, out_channels=self.num_classes, kernel_size=1)
            # regression layers
            self.regs = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1)
            self.w_h_ = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1)

        self.init_weights()
        # self.final_layer = nn.ModuleList(self.final_layer)



    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes,
                                             out_channels=planes,
                                             kernel_size=kernel,
                                             stride=2,
                                             padding=padding,
                                             output_padding=output_padding,
                                             bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.deconv_layers(x)
        out = [[self.hmap(x), self.regs(x), self.w_h_(x)]]
        return out

    def init_weights(self):
        for m in self.deconv_layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.hmap.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, -2.19)
        for m in self.regs.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        for m in self.w_h_.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)