import argparse
import os
from utils import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='dunet', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='0, 1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='DUNet', help='model: DUNet')
        self.parser.add_argument('--pretrained_model', type=str, default='./checkpoints/resnet50-imagenet.pth', help='pretrained_model')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')

        # input/output sizes
        self.parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
        self.parser.add_argument('--fineSize', type=str, default='480,640', help='then crop to this size')
        self.parser.add_argument('--label_nc', type=int, default=21, help='# of input image channels')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='./data/voc_12aug',
                                 help='the root of dataset')
        self.parser.add_argument('--dataset_mode', type=str, default='voc',
                                 help='the root of dataset')
        self.parser.add_argument('--train_list_path', type=str, default='./data/train_aug.txt',
                                 help='the path of train list')
        self.parser.add_argument('--val_list_path', type=str, default='./data/val.txt',
                                 help='the path of val list')
        self.parser.add_argument('--crop_size_h', type=int, default=321,
                                 help='the height of crop size')
        self.parser.add_argument('--crop_size_w', type=int, default=321,
                                 help='the width of crop size')
        self.parser.add_argument('--ignore_label', type=int, default=255,
                                 help='The ignored label')
        self.parser.add_argument('--scale', type=bool, default=True,
                                 help='whether scale?')
        self.parser.add_argument('--mirror', type=bool, default=True,
                                 help='whether mirror?')
        self.parser.add_argument('--max_iters', type=int, default=None,
                                 help='max_iters')

        # # for data augmentation
        # self.parser.add_argument('--flip', action='store_true',help='if specified, flip the images for data argumentation')
        # self.parser.add_argument('--scale', action='store_true',help='if specified, scale the images for data argumentation')
        # self.parser.add_argument('--crop', action='store_true',help='if specified, crop the images for data argumentation')
        # self.parser.add_argument('--colorjitter', action='store_true',help='if specified, crop the images for data argumentation')
        self.parser.add_argument('--inputmode', default='bgr-mean', type=str, help='input image normalize option: bgr-mean, divstd-mean')
        #
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')
        # self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')
        self.parser.add_argument('--verbose', action='store_true', help='if specified, print loss while training')


    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        str_sizes = self.opt.fineSize.split(',')
        self.opt.fineSize = []
        for str_size in str_sizes:
            size_ = int(str_size)
            if size_ >= 0:
                self.opt.fineSize.append(size_)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
