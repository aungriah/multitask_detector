import torch, pdb
import torchvision
import torch.nn.modules

class resnet(torch.nn.Module):
    def __init__(self, config):
        """
        config: hyperparameters of model, including depth of backbone
        returns ResNet of depth indicated in the config file
        """

        num_resnet_layers = config['arch']['backbone_layers']
        pretrained = config['arch']['pretrained_bb']
        super(resnet, self).__init__()
        if num_resnet_layers == '18':
            model = torchvision.models.resnet18(pretrained=pretrained)
        elif num_resnet_layers == '34':
            model = torchvision.models.resnet34(pretrained=pretrained)
        elif num_resnet_layers == '50':
            model = torchvision.models.resnet50(pretrained=pretrained)
        elif num_resnet_layers == '101':
            model = torchvision.models.resnet101(pretrained=pretrained)
        elif num_resnet_layers == '152':
            model = torchvision.models.resnet152(pretrained=pretrained)
        elif num_resnet_layers == '50next':
            model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        elif num_resnet_layers == '101next':
            model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        elif num_resnet_layers == '50wide':
            model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        elif num_resnet_layers == '101wide':
            model = torchvision.models.wide_resnet101_2(pretrained=pretrained)
        else:
            raise NotImplementedError

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.inplanes = model.inplanes

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4