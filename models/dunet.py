import torch.nn as nn
from torch.nn import functional as F
import torch

from models.base_model import *

import shutil
from utils.util import *
from collections import OrderedDict
from tensorboardX import SummaryWriter

affine_par = True
def load_pretrained_model(net, state_dict, strict=True):
    """Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. If :attr:`strict` is ``True`` then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :func:`state_dict()` function.

    Arguments:
        state_dict (dict): A dict containing parameters and
            persistent buffers.
        strict (bool): Strictly enforce that the keys in :attr:`state_dict`
            match the keys returned by this module's `:func:`state_dict()`
            function.
    """
    own_state = net.state_dict()
    # print state_dict.keys()
    # print own_state.keys()
    for name, param in state_dict.items():
        if name in own_state:
            # print name, np.mean(param.numpy())
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            if strict:
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
            else:
                try:
                    own_state[name].copy_(param)
                except Exception:
                    print('Ignoring Error: While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))

        elif strict:
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))
    if strict:
        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class DUpsampling(nn.Module):
    def __init__(self, inplanes, scale, num_class=21, pad=0):
        super(DUpsampling, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, num_class * scale * scale, kernel_size=1, padding = pad,bias=False)
        self.scale = scale
    
    def forward(self, x):
        x = self.conv1(x)
        N, C, H, W = x.size()

        # N, H, W, C
        x_permuted = x.permute(0, 2, 3, 1) 
        x_permuted = x_permuted.contiguous().view((N, H, W * self.scale, int(C / (self.scale))))

        x_permuted = x_permuted.permute(0, 2, 1, 3)
        x_permuted = x_permuted.contiguous().view((N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))

        x = x_permuted.permute(0, 3, 2, 1)
        
        return x
        
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                            padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
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

class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        # self.conv3 = DeformConv(64, 128, (3, 3), stride=1, padding=1, num_deformable_groups=1)
        # self.conv3_deform = conv3x3(64, 2 * 3 * 3)

        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1, 1, 1))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        # input 528 * 528
        x = self.relu1(self.bn1(self.conv1(x)))  # 264 * 264
        x = self.relu2(self.bn2(self.conv2(x)))  # 264 * 264
        x = self.relu3(self.bn3(self.conv3(x)))  # 264 * 264
        
        x_13 = x
        x = self.maxpool(x)  # 66 * 66
        x = self.layer1(x)  # 66 * 66
        x = self.layer2(x)  # 33 * 33
        x = self.layer3(x)  # 66 * 66
        x_46 = x
        x = self.layer4(x)  # 33 * 33

        x_13 = F.interpolate(x_13, [x_46.size()[2],x_46.size()[3]], mode='bilinear', align_corners=True)
        x_low = torch.cat((x_13, x_46), dim=1)
        return x, x_low

class Encoder(nn.Module):
    def __init__(self, pretrain = False, model_path = ' '):
        super(Encoder, self).__init__()
        self.model = ResNet(Bottleneck, [3, 4, 6, 3])
        if pretrain:
            load_pretrained_model(self.model, torch.load(model_path), strict=False)
    def forward(self, x):
        x, x_low = self.model(x)
        return x, x_low

class Decoder(nn.Module):
    def __init__(self, num_class, bn_momentum=0.1):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(1152, 48, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        # self.conv2 = SeparableConv2d(304, 256, kernel_size=3)
        # self.conv3 = SeparableConv2d(256, 256, kernel_size=3)
        self.conv2 = nn.Conv2d(2096, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout(0.1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=1)

        self.dupsample = DUpsampling(256, 16, num_class=21)
        self._init_weight()

    def forward(self, x, low_level_feature):
        low_level_feature = self.conv1(low_level_feature)
        low_level_feature = self.bn1(low_level_feature)
        low_level_feature = self.relu(low_level_feature)
        x_4_cat = torch.cat((x, low_level_feature), dim=1)
        x_4_cat = self.conv2(x_4_cat)
        x_4_cat = self.bn2(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout2(x_4_cat)
        x_4_cat = self.conv3(x_4_cat)
        x_4_cat = self.bn3(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout3(x_4_cat)
        x_4_cat = self.conv4(x_4_cat)

        out = self.dupsample(x_4_cat)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DUNet(nn.Module):
    def __init__(self, encoder_pretrain = False, model_path = ' ', num_class=21):
        super(DUNet, self).__init__()
        self.encoder = Encoder(pretrain=encoder_pretrain, model_path=model_path)
        self.decoder = Decoder(num_class)
    def forward(self, x):
        x, x_low = self.encoder(x)
        x = self.decoder(x, x_low)

        return x


class DUNet_Solver(BaseModel):
    def __init__(self, opt):
        BaseModel.initialize(self, opt)

        self.model = DUNet(encoder_pretrain = True, model_path=self.opt.pretrained_model, num_class=opt.label_nc)

        #self.device =
        if self.opt.isTrain:
            self.criterionSeg = torch.nn.CrossEntropyLoss(ignore_index=255).cuda()
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.opt.lr, momentum=self.opt.momentum,
                                             weight_decay=self.opt.wd)
            self.old_lr = self.opt.lr
            self.averageloss = []

            self.model_path = './models'
            self.data_path = './data'
            shutil.copyfile(os.path.join(self.model_path, 'dunet.py'), os.path.join(self.model_dir, 'dunet.py'))
            shutil.copyfile(os.path.join(self.model_path, 'base_model.py'), os.path.join(self.model_dir, 'base_model.py'))
            self.writer = SummaryWriter(self.tensorborad_dir)
            self.counter = 0

        self.model.cuda()
        self.model = nn.DataParallel(self.model, device_ids=opt.gpu_ids)
        self.normweightgrad=0.

    def forward(self, data, isTrain=True):
        self.model.zero_grad()

        self.image = data[0].cuda()
        self.image.requires_grad = not isTrain


        if data[1] is not None:
            self.seggt = data[1].cuda()
        else:
            self.seggt = None

        input_size = self.image.size()
        self.segpred = self.model(self.image)

        if self.opt.isTrain:
            self.loss = self.criterionSeg(self.segpred, self.seggt.long())
            self.averageloss += [self.loss.data[0]]

        segpred = self.segpred.max(1, keepdim=True)[1]
        self.seggt=torch.unsqueeze(self.seggt, dim=1)

        return self.seggt, segpred

    def backward(self, step, total_step):
        self.loss.backward()
        self.optimizer.step()

        if step % self.opt.iterSize == 0:
            self.update_learning_rate(step, total_step)
            trainingavgloss = np.mean(self.averageloss)
            if self.opt.verbose:
                print ('  Iter: %d, Loss: %f' % (step, trainingavgloss) )
    def freeze_bn(self):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            
    def get_visuals(self, step):
        ############## Display results and errors ############
        if self.opt.isTrain:
            self.trainingavgloss = np.mean(self.averageloss)
            if self.opt.verbose:
                print ('  Iter: %d, Loss: %f' % (step, self.trainingavgloss) )
            self.writer.add_scalar(self.opt.name+'/trainingloss/', self.trainingavgloss, step)
            self.averageloss = []


        return OrderedDict([('image', tensor2im(self.image.data[0], inputmode=self.opt.inputmode)),
                            ('segpred', tensor2label(self.segpred.data[0], self.opt.label_nc)),
                            ('seggt', tensor2label(self.seggt.data[0], self.opt.label_nc))
                            ])

    def update_tensorboard(self, data, step):
        if self.opt.isTrain:
            self.writer.add_scalar(self.opt.name+'/Accuracy/', data[0], step)
            self.writer.add_scalar(self.opt.name+'/Accuracy_Class/', data[1], step)
            self.writer.add_scalar(self.opt.name+'/Mean_IoU/', data[2], step)
            self.writer.add_scalar(self.opt.name+'/FWAV_Accuracy/', data[3], step)

            self.trainingavgloss = np.mean(self.averageloss)
            self.writer.add_scalars(self.opt.name+'/loss', {"train": self.trainingavgloss,
                                                             "val": np.mean(self.averageloss)}, step)

            self.writer.add_scalars('trainingavgloss/', {self.opt.name: self.trainingavgloss}, step)
            self.writer.add_scalars('valloss/', {self.opt.name: np.mean(self.averageloss)}, step)
            self.writer.add_scalars('val_MeanIoU/', {self.opt.name: data[2]}, step)

            file_name = os.path.join(self.save_dir, 'MIoU.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('%f\n' % (data[2]))
            # self.writer.add_scalars('losses/'+self.opt.name, {"train": self.trainingavgloss,
            #                                                  "val": np.mean(self.averageloss)}, step)
            self.averageloss = []

    def save(self, which_epoch):
        # self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.model, 'net', which_epoch, self.gpu_ids)

    def load(self):
        self.load_network(self.model, 'net',self.opt.which_epoch)

    def update_learning_rate(self, step, total_step):

        lr = max(self.opt.lr * ((1 - float(step) / total_step) ** (self.opt.lr_power)), 1e-6)
        # drop_ratio = (1. * float(total_step - step) / (total_step - step + 1)) ** self.opt.lr_power
        # lr = self.old_lr * drop_ratio

        self.writer.add_scalar(self.opt.name+'/Learning_Rate/', lr, step)

        self.optimizer.param_groups[0]['lr'] = lr

        if self.opt.verbose:
            print('     update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
    def name(self):
        return 'DUNet'


if __name__ == '__main__':
    device=torch.device("cuda:0")
    input = torch.rand(1, 3, 528, 528)
    input = input.to(device)
    net = DUNet()
    net.to(device)
    output = net(input)
    print(output.size())
    pass


        
