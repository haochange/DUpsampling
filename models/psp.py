import torch.nn as nn
from torch.nn import functional as F
import torch
from .base_model import *

import shutil
from utils.util import *
from collections import OrderedDict
from tensorboardX import SummaryWriter

affine_par = True

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual      
        out = self.relu_inplace(out)

        return out


class PSPModule(nn.Module):

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self.__make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.Dropout2d(0.1)
        )

    def __make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size = (size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) \
                  for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1,1,1))

        self.head = nn.Sequential(PSPModule(2048, 512),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        # input 512 * 512
        x = self.relu1(self.bn1(self.conv1(x)))  # 256 * 256
        x = self.relu2(self.bn2(self.conv2(x)))  # 256 * 256
        x = self.relu3(self.bn3(self.conv3(x)))  # 256 * 256
        x = self.maxpool(x)                      # 129 * 129
        x = self.layer1(x)                       # 129 * 129
        x = self.layer2(x)                       # 65 * 65
        x = self.layer3(x)                       # 65 * 65
        x_dsn = self.dsn(x)                      # 65 * 65
        x = self.layer4(x)                       # 65 * 65
        x = self.head(x)                         # 65 * 65
        return [x, x_dsn]


def PSP_Res(num_classes=21):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
    return model

class PSP_Solver(BaseModel):
    def __init__(self, opt, dataset=None):
        BaseModel.initialize(self, opt)
        self.model = PSP_Res()
        #self.device =
        if self.opt.isTrain:
            self.criterionSeg = torch.nn.CrossEntropyLoss(ignore_index=255).cuda()
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.opt.lr, momentum=self.opt.momentum,
                                             weight_decay=self.opt.wd)
            self.old_lr = self.opt.lr
            self.averageloss = []

            self.model_path = './models'
            self.data_path = './data'
            shutil.copyfile(os.path.join(self.model_path, 'psp.py'), os.path.join(self.model_dir, 'psp.py'))
            shutil.copyfile(os.path.join(self.model_path, 'base_model.py'), os.path.join(self.model_dir, 'base_model.py'))
            self.writer = SummaryWriter(self.tensorborad_dir)
            self.counter = 0

        if not self.isTrain or self.opt.continue_train:
            if self.opt.pretrained_model!='':
                self.load_pretrained_network(self.model, self.opt.pretrained_model, self.opt.which_epoch, strict=False)
                print("Successfully loaded from pretrained model with given path!")
            else:
                self.load()
                print("Successfully loaded model, continue training....!")
        self.model.cuda()
        self.normweightgrad=0.

if __name__ == '__main__':
    device=torch.device("cuda:0")
    input = torch.rand(2, 3, 512, 512)
    input = input.to(device)
    net = PSP_Res()
    net.to(device)
    output = net(input)
    print(output[1].cpu().data)
    pass
