import torch
import torch.nn as nn
import torchvision


from torchinfo import summary


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 7x7 convolution
def conv7x7(in_channels, out_channels, stride=2):
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, 
                     stride=stride, padding=3, bias=False)

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# 1x1 convolution
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                     stride=stride, padding=0, bias=False)


# Residual block using 3x3->3x3 kernel
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# Residual block using 1x1->3x3->1x1 kernel
class ResidualBlockBottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlockBottleneck, self).__init__()
        self.mid_channels = int(out_channels/4)

        self.conv1 = conv1x1(in_channels, self.mid_channels)
        self.bn1 = nn.BatchNorm2d(self.mid_channels)
        self.conv2 = conv3x3(self.mid_channels, self.mid_channels, stride)
        self.bn2 = nn.BatchNorm2d(self.mid_channels )
        self.conv3 = conv1x1(self.mid_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        nn.init.normal_(self.conv3.weight, 0, 0.01)

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
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet for 18 and 34
#    18: layers = [2,2,2,2]
#    34: layers = [3,4,6,3]
class ResNet_small(nn.Module):
    def __init__(self, block, layers):
        super(ResNet_small, self).__init__()
        self.in_channels = 64

        self.conv1 = conv7x7(3, 64)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_layer = self.make_layer(block, 64, layers[0])
        self.conv3_layer = self.make_layer(block, 128, layers[1], 2)
        self.conv4_layer = self.make_layer(block, 256, layers[2], 2)
        self.conv5_layer = self.make_layer(block, 512, layers[3], 2)

        nn.init.normal_(self.conv1.weight, 0, 0.01)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))

        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.max_pool(out)
        out = self.conv2_layer(out)
        out = self.conv3_layer(out)
        out = self.conv4_layer(out)
        out = self.conv5_layer(out)
        return out

# ResNet for 50, 101 and 152
#    50:  layers = [3,4,6,3]
#    101: layers = [3,4,23,3]
#    152: layers = [3,8,36,3]
class ResNet_large(nn.Module):
    def __init__(self, block, layers):
        super(ResNet_large, self).__init__()
        self.in_channels = 64

        self.conv1 = conv7x7(3, 64)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_layer = self.make_layer(block, 256, layers[0])
        self.conv3_layer = self.make_layer(block, 512, layers[1], 2)
        self.conv4_layer = self.make_layer(block, 1024, layers[2], 2)
        self.conv5_layer = self.make_layer(block, 2048, layers[3], 2)

        nn.init.normal_(self.conv1.weight, 0, 0.01)

    def make_layer(self, block, out_channels, blocks, stride=1):
        #downsample = None
        #if stride != 1:
        #    downsample = nn.Sequential(
        #        conv1x1(self.in_channels, out_channels, stride=stride),
        #        nn.BatchNorm2d(out_channels))

        downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))

        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.max_pool(out)
        out = self.conv2_layer(out)
        out = self.conv3_layer(out)
        out = self.conv4_layer(out)
        out = self.conv5_layer(out)
        return out


def test():
    resnet_50 = ResNet_large(ResidualBlockBottleneck, [3, 4, 6, 3]).to(device)
    summary(resnet_50)
    
    model = torchvision.models.resnet50().to(device)
    summary(model)

    data = torch.rand(1, 3, 32, 32).cuda()
    labels = torch.rand(1, 1000).cuda()


    #prediction = resnet_50(data) # forward pass
    prediction = model(data) # forward pass
    print("pred: ", prediction.shape)




if __name__ == "__main__":
    test()

