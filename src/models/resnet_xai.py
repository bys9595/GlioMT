import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from timm.models._builder import build_model_with_cfg
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models._registry import generate_default_cfgs

from .resnet_utils.resnet_layers import *

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed3d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.clone = Clone()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.act1 = ReLU(inplace=True)
        self.act2 = ReLU(inplace=True)

        self.add = Add()

        self.register_forward_hook(forward_hook)

    def forward(self, x):
        x1, x2 = self.clone(x, 2)

        out = self.conv1(x1)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x2 = self.downsample(x2)

        out = self.add([out, x2])
        out = self.act2(out)

        return out

    def relprop(self, R, alpha):
        out = self.act2.relprop(R, alpha)
        out, x2 = self.add.relprop(out, alpha)

        if self.downsample is not None:
            x2 = self.downsample.relprop(x2, alpha)

        out = self.bn2.relprop(out, alpha)
        out = self.conv2.relprop(out, alpha)

        out = self.act1.relprop(out, alpha)
        out = self.bn1.relprop(out, alpha)
        x1 = self.conv1.relprop(out, alpha)

        return self.clone.relprop([x1, x2], alpha)

    def m_relprop(self, R, pred, alpha):
        out = self.act2.m_relprop(R, pred, alpha)
        out, x2 = self.add.m_relprop(out, pred, alpha)

        if self.downsample is not None:
            x2 = self.downsample.m_relprop(x2, pred, alpha)

        out = self.bn2.m_relprop(out, pred, alpha)
        out = self.conv2.m_relprop(out, pred, alpha)

        out = self.act1.m_relprop(out, pred, alpha)
        out = self.bn1.m_relprop(out, pred, alpha)
        x1 = self.conv1.m_relprop(out, pred, alpha)

        return self.clone.m_relprop([x1, x2], pred, alpha)
    
    def RAP_relprop(self, R):
        out = self.act2.RAP_relprop(R)
        out, x2 = self.add.RAP_relprop(out)

        if self.downsample is not None:
            x2 = self.downsample.RAP_relprop(x2)

        out = self.bn2.RAP_relprop(out)
        out = self.conv2.RAP_relprop(out)

        out = self.act1.RAP_relprop(out)
        out = self.bn1.RAP_relprop(out)
        x1 = self.conv1.RAP_relprop(out)

        return self.clone.RAP_relprop([x1, x2])

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.clone = Clone()

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2d(planes)
        self.act1 = ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = BatchNorm2d(planes)
        self.act2 = ReLU(inplace=True)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = BatchNorm2d(planes * self.expansion)
        self.act3 = ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.add = Add()

        self.register_forward_hook(forward_hook)

    def forward(self, x):
        # x1, x2 = self.clone(x, 2)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            x = self.downsample(x)

        # out = self.add([out, x2])
        out = self.add([out, x])
        out = self.act3(out)

        return out

    def relprop(self, R, alpha):
        out = self.act3.relprop(R, alpha)

        out, x = self.add.relprop(out, alpha)

        if self.downsample is not None:
            x = self.downsample.relprop(x, alpha)

        out = self.bn3.relprop(out, alpha)
        out = self.conv3.relprop(out, alpha)

        out = self.act2.relprop(out, alpha)
        out = self.bn2.relprop(out, alpha)
        out = self.conv2.relprop(out, alpha)

        out = self.act1.relprop(out, alpha)
        out = self.bn1.relprop(out, alpha)
        x1 = self.conv1.relprop(out, alpha)

        return x1 + x
        # return self.clone.relprop([x1, x2], alpha)
    def m_relprop(self, R, pred, alpha):
        out = self.act3.m_relprop(R, pred, alpha)

        out, x = self.add.m_relprop(out, pred, alpha)

        if self.downsample is not None:
            x = self.downsample.m_relprop(x, pred, alpha)

        out = self.bn3.m_relprop(out, pred, alpha)
        out = self.conv3.m_relprop(out, pred, alpha)

        out = self.act2.m_relprop(out, pred, alpha)
        out = self.bn2.m_relprop(out, pred, alpha)
        out = self.conv2.m_relprop(out, pred, alpha)

        out = self.act1.m_relprop(out, pred, alpha)
        out = self.bn1.m_relprop(out, pred, alpha)
        x1 = self.conv1.m_relprop(out, pred, alpha)
        if torch.is_tensor(x1) == True:
            return x1 + x
        else:
            for i in range(len(x1)):
                x1[i] = x1[i] + x[i]
            return x1

    def RAP_relprop(self, R):
        out = self.act3.RAP_relprop(R)

        out, x = self.add.RAP_relprop(out)

        if self.downsample is not None:
            x = self.downsample.RAP_relprop(x)

        out = self.bn3.RAP_relprop(out)
        out = self.conv3.RAP_relprop(out)

        out = self.act2.RAP_relprop(out)
        out = self.bn2.RAP_relprop(out)
        out = self.conv2.RAP_relprop(out)

        out = self.act1.RAP_relprop(out)
        out = self.bn1.RAP_relprop(out)
        x1 = self.conv1.RAP_relprop(out)

        return x1 + x


class ResNet(nn.Module):
    
    def __init__(self, block, layers, num_classes=1000, in_chans=1, base_ch=64, long= False, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = base_ch
        self.conv1 = Conv2d(in_chans, base_ch, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d(base_ch)
        self.act1 = ReLU(inplace=True)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, base_ch, layers[0])
        self.layer2 = self._make_layer(block, base_ch*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_ch*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_ch*8, layers[3], stride=2)
        self.global_pool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(base_ch*8 * block.expansion, num_classes)
        self.long = long
        self.num_classes = num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)

    def CLRP(self, x, maxindex = [None]):
        if maxindex == [None]:
            maxindex = torch.argmax(x, dim=1)
        R = torch.ones(x.shape).cuda()
        R /= -self.num_classes
        for i in range(R.size(0)):
            R[i, maxindex[i]] = 1
        return R

    def forward(self, x, return_inter=False):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.global_pool(layer4)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        
        
        if return_inter:
            return [layer1, layer2, layer3, layer4, z]
        else:
            return z

    def relprop(self, R, alpha, flag = 'inter'):
        if self.long:
            R = self.fc.relprop(R, alpha)
            R = R.reshape_as(self.global_pool.Y)
            R = self.global_pool.relprop(R, alpha)
            R = self.layer4.relprop(R, alpha)
            R = self.layer3.relprop(R, alpha)
            R = self.layer2.relprop(R, alpha)
            R = self.layer1.relprop(R, alpha)
            R = self.maxpool.relprop(R, alpha)
            R = self.act1.relprop(R, alpha)
            R = self.bn1.relprop(R, alpha)
            R = self.conv1.relprop(R, alpha)
        else:
            R = self.fc.relprop(R, alpha)
            R = R.reshape_as(self.global_pool.Y)
            R = self.global_pool.relprop(R, alpha)
            if flag == 'layer4': return R
            R = self.layer4.relprop(R, alpha)
            if flag == 'layer3': return R
            R = self.layer3.relprop(R, alpha)
            if flag == 'layer2': return R
            R = self.layer2.relprop(R, alpha)
            if flag == 'layer1': return R

        return R

    def m_relprop(self, R, pred, alpha):
        R = self.fc.m_relprop(R, pred, alpha)
        if torch.is_tensor(R) == False:
            for i in range(len(R)):
                R[i] = R[i].reshape_as(self.global_pool.Y)
        else:
            R = R.reshape_as(self.global_pool.Y)
        R = self.global_pool.m_relprop(R, pred, alpha)

        R = self.layer4.m_relprop(R, pred, alpha)
        R = self.layer3.m_relprop(R, pred, alpha)
        R = self.layer2.m_relprop(R, pred, alpha)
        R = self.layer1.m_relprop(R, pred, alpha)

        R = self.maxpool.m_relprop(R, pred, alpha)
        R = self.act1.m_relprop(R, pred, alpha)
        R = self.bn1.m_relprop(R, pred, alpha)
        R = self.conv1.m_relprop(R, pred, alpha)

        return R

    def RAP_relprop(self, R):
        R = self.fc.RAP_relprop(R)
        R = R.reshape_as(self.global_pool.Y)
        R = self.global_pool.RAP_relprop(R)

        R = self.layer4.RAP_relprop(R)
        R = self.layer3.RAP_relprop(R)
        R = self.layer2.RAP_relprop(R)
        R = self.layer1.RAP_relprop(R)

        R = self.maxpool.RAP_relprop(R)
        R = self.act1.RAP_relprop(R)
        R = self.bn1.RAP_relprop(R)
        R = self.conv1.RAP_relprop(R)

        return R
    


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }



default_cfgs = generate_default_cfgs({
    # torchvision resnet weights
    'resnet18.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet18-5c106cde.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet34.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet50.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet50-19c8e357.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet50.tv2_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet101.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet101.tv2_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet101-cd907fc2.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet152.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet152.tv2_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet152-f82ba261.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),

})


def create_model(model_name, pretrained=False, **kwargs):    
    return globals()[model_name](pretrained=pretrained, **kwargs)


def _create_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ResNet, variant, pretrained, **kwargs)


def resnet18(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-18 model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2])
    return _create_resnet('resnet18', pretrained, **dict(model_args, **kwargs))


def resnet34(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-34 model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3])
    return _create_resnet('resnet34', pretrained, **dict(model_args, **kwargs))


def resnet26(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-26 model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 2])
    return _create_resnet('resnet26', pretrained, **dict(model_args, **kwargs))


def resnet50(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_resnet('resnet50', pretrained, **dict(model_args, **kwargs))


def resnet101(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3])
    return _create_resnet('resnet101', pretrained, **dict(model_args, **kwargs))

def resnet152(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-152 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3])
    return _create_resnet('resnet152', pretrained, **dict(model_args, **kwargs))


def resnet200(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-200 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 24, 36, 3])
    return _create_resnet('resnet200', pretrained, **dict(model_args, **kwargs))



if __name__ == '__main__':
    import torch
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    
    a = torch.ones((1, 4, 96, 96, 96)).cuda().requires_grad_(True)
    model = resnet50().cuda()
    
    outputs = model(a, mode='layer3', target_class=[2])
    cc = 1
    
    