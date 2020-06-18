import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
import torch.utils.model_zoo as modelzoo


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## Last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Shape of a Convolutional Layer
        # K - out_channels : the number of filters in the convolutional layer
        # F - kernel_size
        # S - the stride of the convolution
        # P - the padding
        # W - the width/height (square) of the previous layer
        
        # Since there are F*F*D weights per filter
        # The total number of weights in the convolutional layer is K*F*F*D
        
        # 224 by 224 pixels
        
        ## self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        # output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (1, 220, 220)
        # after one pool layer, this becomes (10, 13, 13)
        self.conv1 = nn.Conv2d(3, 32, 5).cuda(device)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2).cuda(device)
        
        # 220/2 = 110
        # output size = (W-F)/S +1 = (110-3)/1 + 1 = 108
        # the output Tensor for one image, will have the dimensions: (32, 110, 110)
        self.conv2 = nn.Conv2d(32, 64, 3).cuda(device)
        
        # output size = (W-F)/S +1 = (54-3)/1 + 1 = 52
        # the output Tensor for one image, will have the dimensions: (64, 54, 54)
        self.conv3 = nn.Conv2d(64, 128, 3).cuda(device)
        
        # output size = (W-F)/S +1 = (26-3)/1 + 1 = 24
        # the output Tensor for one image, will have the dimensions: (128, 26, 26)
        self.conv4 = nn.Conv2d(128, 256, 3).cuda(device)
        
        # output size = (W-F)/S +1 = (12-3)/1 + 1 = 10
        # the output Tensor for one image, will have the dimensions: (256, 12, 12)
        self.conv5 = nn.Conv2d(256, 512, 1).cuda(device)
        
        # output size = (W-F)/S +1 = (6-1)/1 + 1 = 6
        # the output Tensor for one image, will have the dimensions: (512, 6, 6)
        
        # Fully-connected (linear) layers
        self.fc1 = nn.Linear(512*15*15, 1024).cuda(device)
        self.fc2 = nn.Linear(1024, 512).cuda(device)
        self.fc3 = nn.Linear(512, 68*2).cuda(device)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.25).cuda(device)
        
        
    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        # 5 conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        
        # Prep for linear layer / Flatten
        x = x.view(x.size(0), -1)
        
        # linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        
        
        return x


#BUilding Resnet From Scratch


resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
                )

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out


def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum-1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)
        self.init_weight()

        # Fully-connected (linear) layers
        self.fc1 = nn.Linear(512*6*6, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 68*2)
        # Dropout
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x) # 1/8
        feat16 = self.layer3(feat8) # 1/16
        feat32 = self.layer4(feat16) # 1/32

        # Prep for linear layer / Flatten
        out = feat32.view(feat32.size(0), -1)
        # linear layers with dropout in between
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)

        return feat8, feat16, feat32, out

    def init_weight(self):
        state_dict = modelzoo.load_url(resnet18_url)
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if 'fc' in k: continue
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module,  nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class Resnet18Plus(nn.Module):
    def __init__(self):
        super(Resnet18Plus, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)
        self.init_weight()

        # Fully-connected (linear) layers
        self.fc1 = nn.Linear(512*16*16, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 68*2)
        # Dropout
        self.dropout = nn.Dropout(p=0.25)

        # Fully-connected (linear) layers
        self.fc1b = nn.Linear(512*16*16, 1024)
        self.fc2b = nn.Linear(1024, 512)
        self.fc3b = nn.Linear(512, 68*2)
        # Dropout
        self.dropoutb = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x) # 1/8
        feat16 = self.layer3(feat8) # 1/16
        feat32 = self.layer4(feat16) # 1/32

        # Prep for linear layer / Flatten
        outF = feat32.view(feat32.size(0), -1)
        # linear layers with dropout in between
        out = F.relu(self.fc1(outF))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)

        outb = F.relu(self.fc1(outF))
        outb = self.dropout(outb)
        outb = F.relu(self.fc2(outb))
        outb = self.dropout(outb)
        outb = self.fc3(outb)
        del outF

        return out,outb

    def init_weight(self):
        state_dict = modelzoo.load_url(resnet18_url)
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if 'fc' in k: continue
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module,  nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params



class NetReg(nn.Module):

    def __init__(self):
        super(NetReg, self).__init__()
        
        ## Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## Last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Shape of a Convolutional Layer
        # K - out_channels : the number of filters in the convolutional layer
        # F - kernel_size
        # S - the stride of the convolution
        # P - the padding
        # W - the width/height (square) of the previous layer
        
        # Since there are F*F*D weights per filter
        # The total number of weights in the convolutional layer is K*F*F*D
        
        # 512 by 512 pixels
        
        ## self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        # output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (1, 220, 220)
        # after one pool layer, this becomes (10, 13, 13)
        self.conv1 = nn.Conv2d(3, 32, 5).cuda(device)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2).cuda(device)
        
        # 220/2 = 110
        # output size = (W-F)/S +1 = (110-3)/1 + 1 = 108
        # the output Tensor for one image, will have the dimensions: (32, 110, 110)
        self.conv2 = nn.Conv2d(32, 64, 3).cuda(device)
        
        # output size = (W-F)/S +1 = (54-3)/1 + 1 = 52
        # the output Tensor for one image, will have the dimensions: (64, 54, 54)
        self.conv3 = nn.Conv2d(64, 128, 3).cuda(device)
        
        # output size = (W-F)/S +1 = (26-3)/1 + 1 = 24
        # the output Tensor for one image, will have the dimensions: (128, 26, 26)
        self.conv4 = nn.Conv2d(128, 256, 3).cuda(device)
        
        # output size = (W-F)/S +1 = (12-3)/1 + 1 = 10
        # the output Tensor for one image, will have the dimensions: (256, 12, 12)
        self.conv5 = nn.Conv2d(256, 512, 1).cuda(device)
        
        # output size = (W-F)/S +1 = (6-1)/1 + 1 = 6
        # the output Tensor for one image, will have the dimensions: (512, 6, 6)
        
        # Fully-connected (linear) layers
        self.fc1 = nn.Linear(512*15*15, 1024).cuda(device)
        self.fc2 = nn.Linear(1024, 512).cuda(device)
        self.fc3 = nn.Linear(512, 68*2).cuda(device)

        # Fully-connected (linear) layers
        self.fc1b = nn.Linear(512*15*15, 1024).cuda(device)
        self.fc2b = nn.Linear(1024, 512).cuda(device)
        self.fc3b = nn.Linear(512, 68*2).cuda(device)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.25).cuda(device)
        self.dropoutb = nn.Dropout(p=0.25).cuda(device)

        
        
    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        # 5 conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        
        # Prep for linear layer / Flatten and second regressor
        x = x.view(x.size(0), -1)
        x2 = x.clone()
        
        # linear layers with dropout in between for x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        # linear layers with dropout in between for x2 which is hair

        x2 = F.relu(self.fc1b(x2))
        x2 = self.dropoutb(x2)
        x2 = F.relu(self.fc2b(x2))
        x2 = self.dropoutb(x2)
        x2 = self.fc3b(x2)
        
        return x,x2
