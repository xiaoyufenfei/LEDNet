import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def channel_shuffle(x,groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    #reshape
    x = x.view(batchsize,groups,
        channels_per_group,height,width)

    x = torch.transpose(x,1,2).contiguous()

    #flatten
    x = x.view(batchsize,-1,height,width)

    return x


class Conv2dBnRelu(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=1,padding=0,dilation=1,bias=True):
        super(Conv2dBnRelu,self).__init__()

        self.conv = nn.Sequential(
		nn.Conv2d(in_ch,out_ch,kernel_size,stride,padding,dilation=dilation,bias=bias),
		nn.BatchNorm2d(out_ch, eps=1e-3),
		nn.ReLU(inplace=True)
	)

    def forward(self, x):
        return self.conv(x)


##after Concat -> BN, you also can use Dropout like SS_nbt_module may be make a good result!
class DownsamplerBlock (nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel-in_channel, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
		output = self.relu(output)
        return output


class SS_nbt_module(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        oup_inc = chann//2

        #dw
        self.conv3x1_1_l = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1_l = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1_l = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.conv3x1_2_l = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2_l = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1,dilated))

        self.bn2_l = nn.BatchNorm2d(oup_inc, eps=1e-03)

        #dw
        self.conv3x1_1_r = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1_r = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1_r = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.conv3x1_2_r = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2_r = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1,dilated))

        self.bn2_r = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropprob)

    @staticmethod
    def _concat(x,out):
        return torch.cat((x,out),1)

    def forward(self, input):

        x1 = input[:,:(input.shape[1]//2),:,:]
        x2 = input[:,(input.shape[1]//2):,:,:]

        output1 = self.conv3x1_1_l(x1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_1_l(output1)
        output1 = self.bn1_l(output1)
        output1 = self.relu(output1)

        output1 = self.conv3x1_2_l(output1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_2_l(output1)
        output1 = self.bn2_l(output1)


        output2 = self.conv1x3_1_r(x2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_1_r(output2)
        output2 = self.bn1_r(output2)
        output2 = self.relu(output2)

        output2 = self.conv1x3_2_r(output2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_2_r(output2)
        output2 = self.bn2_r(output2)

        if (self.dropout.p != 0):
            output1 = self.dropout(output1)
            output2 = self.dropout(output2)

        out = self._concat(output1,output2)
        out = F.relu(input+out,inplace=True)
        return channel_shuffle(out,2)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_block = DownsamplerBlock(3,32)

        self.layers = nn.ModuleList()

        for x in range(0, 3):
            self.layers.append(SS_nbt_module(32, 0.03, 1))


        self.layers.append(DownsamplerBlock(32,64))


        for x in range(0, 2):
            self.layers.append(SS_nbt_module(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 1):
            self.layers.append(SS_nbt_module(128, 0.3, 1))
            self.layers.append(SS_nbt_module(128, 0.3, 2))
            self.layers.append(SS_nbt_module(128, 0.3, 5))
            self.layers.append(SS_nbt_module(128, 0.3, 9))

        for x in range(0, 1):
            self.layers.append(SS_nbt_module(128, 0.3, 2))
            self.layers.append(SS_nbt_module(128, 0.3, 5))
            self.layers.append(SS_nbt_module(128, 0.3, 9))
            self.layers.append(SS_nbt_module(128, 0.3, 17))


    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        return output


class Features(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.extralayer1 = nn.MaxPool2d(2, stride=2)
        self.extralayer2 = nn.AvgPool2d(14,1,0)

    def forward(self, input):
        output = self.encoder(input)
        output = self.extralayer1(output)
        output = self.extralayer2(output)
        return output

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.linear = nn.Linear(128, num_classes)

    def forward(self, input):
        output = input.view(input.size(0), 128) #first is batch_size
        output = self.linear(output)
        return output

class LEDNet(nn.Module):
    def __init__(self, num_classes):  #use encoder to pass pretrained encoder
        super().__init__()

        self.features = Features()
        self.classifier = Classifier(num_classes)

    def forward(self, input):
        output = self.features(input)
        output = self.classifier(output)
        return output

