# coding: utf-8

#############################################
# Consistent Cumulative Logits with ResNet-34
#############################################

# Imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import os
from os import listdir
from os.path import isfile, join
import csv

from torchvision import transforms
from PIL import Image

torch.backends.cudnn.deterministic = True

##########################
# SETTINGS
##########################

# Hyperparameters
learning_rate = 0.0005
num_epochs = 200

# Architecture
NUM_CLASSES = 4
BATCH_SIZE = 256
GRAYSCALE = False

##########################
# MODEL
##########################
DEVICE = torch.device('cpu')

print('something')
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.num_classes = num_classes
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1, padding=2)
        self.fc = nn.Linear(2048 * block.expansion, 1, bias=False)
        self.linear_1_bias = nn.Parameter(torch.zeros(self.num_classes-1).float())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)
        return logits, probas


def resnet34(num_classes, grayscale):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=BasicBlock,
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes,
                   grayscale=grayscale)
    return model


#######################
### Initialize Model
#######################

model = resnet34(NUM_CLASSES, GRAYSCALE)
print('test')
model.load_state_dict(torch.load('model/model.pt', map_location=DEVICE))
model.eval()


############################
### open output files to write
############################
with open('configuration.json', 'r') as configfile:
    config_info=json.load(configfile)
print(config_info['output-files-directory'])
output_file_path=config_info['output-files-directory']
filename_below9=output_file_path+'below-9.txt'
if os.path.exists(filename_below9):
    os.remove(filename_below9)

f_filename_below9=open(filename_below9, "a")

filename_9_13=output_file_path+'range-9-13.txt'
if os.path.exists(filename_9_13):
    os.remove(filename_9_13)

f_filename_9_13=open(filename_9_13, "a")

filename_14_17=output_file_path+'range-14-17.txt'
if os.path.exists(filename_14_17):
    os.remove(filename_14_17)

f_filename_14_17=open(filename_14_17, "a")

filename_above_18=output_file_path+'above-18.txt'
if os.path.exists(filename_above_18):
    os.remove(filename_above_18)

f_filename_above_18=open(filename_above_18, "a")
############################
### Load image from directory and estimate age
############################
cropped_images_path=config_info['cropped-faces-directory']
cropped_files = [f for f in listdir(cropped_images_path) if isfile(join(cropped_images_path, f))]
#print(cropped_files)
with open(cropped_images_path+'cropped-images-details.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
    	min_age=100000
    	original_image_name=row[0]
    	length=len(row)
    	for i in range(1,length):
        	image = Image.open(cropped_images_path+row[i]).convert('RGB')
        	custom_transform = transforms.Compose([transforms.Resize((128, 128)),transforms.CenterCrop((120, 120)),transforms.ToTensor()])
        	image = custom_transform(image)
        	#DEVICE = torch.device('cpu')
        	image = image.to(DEVICE)
        	image = image.unsqueeze(0)
        	with torch.set_grad_enabled(False):
        		logits, probas = model(image)
        		predict_levels = probas > 0.5
        		predicted_label = torch.sum(predict_levels, dim=1)
        		predicted_age=predicted_label.item()
        	if predicted_age < min_age:
        		min_age=predicted_age
    	if min_age == 0:
    		f_filename_below9.write(original_image_name+'\n')
    	elif min_age == 1:
    		f_filename_9_13.write(original_image_name+'\n')
    	elif min_age == 2:
    		f_filename_14_17.write(original_image_name+'\n')
    	else: 	
    		f_filename_above_18.write(original_image_name+'\n')

f_filename_below9.close
f_filename_9_13.close
f_filename_14_17.close
f_filename_above_18.close
