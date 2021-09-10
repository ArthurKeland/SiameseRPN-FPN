import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from lib.custom_transforms import ToTensor

from torchvision.models import alexnet
from torch.autograd import Variable
from torch import nn

from IPython import embed
from .config import config


class SiameseAlexNet(nn.Module):
    def __init__(self, ):
        super(SiameseAlexNet, self).__init__()
        self.featureExtract = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=2), #0
            nn.BatchNorm2d(96), #1
            nn.MaxPool2d(3, stride=2), #2
            nn.ReLU(inplace=True), #3
            nn.Conv2d(96, 256, 5), #4
            nn.BatchNorm2d(256), #5
            nn.MaxPool2d(3, stride=2), #6
            nn.ReLU(inplace=True), #7
            nn.Conv2d(256, 384, 3), #8
            nn.BatchNorm2d(384), #9
            nn.ReLU(inplace=True), #10
            nn.Conv2d(384, 384, 3), #11
            nn.BatchNorm2d(384), #12
            nn.ReLU(inplace=True), #13
            nn.Conv2d(384, 256, 3), #14
            nn.BatchNorm2d(256), #15
        )
        self.anchor_num = config.anchor_num
        self.input_size = config.instance_size
        self.score_displacement = int((self.input_size - config.exemplar_size) / config.total_stride)

        self.refine_feature = nn.Conv2d(1376,256,kernel_size=1)

        self.conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.conv_r1 = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)

        self.conv_cls2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.conv_r2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)

        self.regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)

    def forward(self, template, detection):
        N = template.size(0)

        template_feature = self.featureExtract(template)
        detection_feature = self.featureExtract(detection)

        kernel_score = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        kernel_regression = self.conv_r1(template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        conv_score = self.conv_cls2(detection_feature)
        conv_regression = self.conv_r2(detection_feature)

        conv_scores = conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        score_filters = kernel_score.reshape(-1, 256, 4, 4)
        pred_score = F.conv2d(conv_scores, score_filters, groups=N).reshape(N, 10, self.score_displacement + 1,
                                                                            self.score_displacement + 1)

        conv_reg = conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        reg_filters = kernel_regression.reshape(-1, 256, 4, 4)
        pred_regression = self.regress_adjust(
            F.conv2d(conv_reg, reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                              self.score_displacement + 1))
        return pred_score, pred_regression

    def track_init(self, template):
        N = template.size(0)

        x = template
        for name, module in self.featureExtract.named_children():
            x = module(x)
            if name == '3': # conv1
                conv1 = x
            elif name == '7':
                conv2 = x
            elif name == '10':
                conv3 = x
            elif name == '13':
                conv4 = x
            elif name == '15':
                conv5 = x 
        b,c,w,h = conv5.shape

        conv1_resize = F.interpolate(conv1,(w,h))
        conv2_resize = F.interpolate(conv2,(w,h))
        conv3_resize = F.interpolate(conv3,(w,h))
        conv4_resize = F.interpolate(conv4,(w,h))
        
        ''' TODO'''
        # template_feature = torch.cat((conv1_resize,conv2_resize,conv3_resize,conv4_resize,conv5),dim=1)
        # template_feature = self.refine_feature(template_feature)


        # template_feature  = conv5 #baseline
        template_feature  = 0.1*(conv2_resize) + conv5 # 2+5 (the best performance)
        # template_feature  = 0.1*(conv2_resize + conv3_resize[:,:256,:,:]) + conv5 # 2+3+5
        # template_feature  = 0.1*(conv2_resize + conv3_resize[:,:256,:,:] + conv4_resize[:,:256,:,:]) + conv5 # 2+3+4+5

        kernel_score = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num, template_feature.shape[1], 4, 4)
        kernel_regression = self.conv_r1(template_feature).view(N, 4 * self.anchor_num, template_feature.shape[1], 4, 4)
        self.score_filters = kernel_score.reshape(-1, template_feature.shape[1], 4, 4)
        self.reg_filters = kernel_regression.reshape(-1, template_feature.shape[1], 4, 4)

    def track(self, detection):
        N = detection.size(0)
        x = detection
        for name, module in self.featureExtract.named_children():
            x = module(x)
            if name == '3': # conv1
                conv1 = x
            elif name == '7':
                conv2 = x
            elif name == '10':
                conv3 = x
            elif name == '13':
                conv4 = x
            elif name == '15':
                conv5 = x 
        b,c,w,h = conv5.shape

        conv1_resize = F.interpolate(conv1,(w,h))
        conv2_resize = F.interpolate(conv2,(w,h))
        conv3_resize = F.interpolate(conv3,(w,h))
        conv4_resize = F.interpolate(conv4,(w,h))

        # detection_feature = torch.cat((conv1_resize,conv2_resize,conv3_resize,conv4_resize,conv5),dim=1)
        # detection_feature = self.refine_feature(template_feature)

        # detection_feature  = conv5 #baseline
        detection_feature  = 0.1*(conv2_resize) + conv5 # 2+5 (the best performance)
        # detection_feature  = 0.1*(conv2_resize + conv3_resize[:,:256,:,:]) + conv5 # 2+3+5
        # detection_feature  = 0.1*(conv2_resize + conv3_resize[:,:256,:,:] + conv4_resize[:,:256,:,:]) + conv5 # 2+3+4+5

        conv_score = self.conv_cls2(detection_feature)
        conv_regression = self.conv_r2(detection_feature)

        conv_scores = conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        pred_score = F.conv2d(conv_scores, self.score_filters, groups=N).reshape(N, 10, self.score_displacement + 1,
                                                                                 self.score_displacement + 1)
        conv_reg = conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        pred_regression = self.regress_adjust(
            F.conv2d(conv_reg, self.reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                   self.score_displacement + 1))
        return pred_score, pred_regression
