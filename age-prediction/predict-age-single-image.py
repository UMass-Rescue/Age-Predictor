from torch_mtcnn import detect_faces
from PIL import Image
from os import listdir
from os.path import isfile, join
import json
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse


from torchvision import transforms
from PIL import Image


#############################
# parse arguments 
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image_path',
                    type=str,
                    required=True)

args = parser.parse_args()
IMAGE_PATH = args.image_path

##########################
# SETTINGS
##########################
# Architecture
NUM_CLASSES = 4
GRAYSCALE = False

##########################
# MODEL
##########################
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#print('something')
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

model = resnet34(NUM_CLASSES, GRAYSCALE)
#print('test')
model.load_state_dict(torch.load('model/model.pt', map_location=DEVICE))
model.eval()

def predict(img):
    custom_transform = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.CenterCrop((120, 120)),
                                       transforms.ToTensor()])
    image = custom_transform(img)
    DEVICE = torch.device('cpu')
    image = image.to(DEVICE)
    image = image.unsqueeze(0)
    with torch.set_grad_enabled(False):
        logits, probas = model(image)
        predict_levels = probas > 0.5
        predicted_label = torch.sum(predict_levels, dim=1)
        predicted_age=predicted_label.item()
        if predicted_age == 0:
        	print('Predicted age class : below 9')
        elif predicted_age == 1:
        	print('Predicted age class : 9 - 13')
        elif predicted_age == 2:
        	print('Predicted age class : 14 - 17')
        else:
         	print('Predicted age class : 18 or above 18')
    return predicted_age

#### Do the actual prediction

#input_image='test.jfif'
image = Image.open(IMAGE_PATH).convert('RGB')
bounding_boxes, landmarks = detect_faces(image)
print("No of faces in image", len(bounding_boxes))
for i in range(len(bounding_boxes)):
    print('Bounding box and age for face '+str(i+1))
    area = (bounding_boxes[i][0],bounding_boxes[i][1],bounding_boxes[i][2],bounding_boxes[i][3])
    cropped_image=image.crop(area)
    print(bounding_boxes[0])
    predict(cropped_image)