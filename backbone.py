import torch
import torch.nn as nn

class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 padding=0,
                 stride=1,
                 dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=False)
        )

    def forward(self, x):
        return self.convs(x)

class ConvBnLeakyReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1):

        super(ConvBnLeakyReLU, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv(x)))

class YOLOv3TinyBackbone(nn.Module):
    def __init__(self, num_classes=1000):
        super(YOLOv3TinyBackbone, self).__init__()
        
        self.stage1 = ConvBnLeakyReLU(3, 16)
        self.stage2 = nn.Sequential(nn.MaxPool2d(2), ConvBnLeakyReLU(16, 32))
        self.stage3 = nn.Sequential(nn.MaxPool2d(2), ConvBnLeakyReLU(32, 64))
        self.stage4 = nn.Sequential(nn.MaxPool2d(2), ConvBnLeakyReLU(64, 128))
        self.stage5 = nn.Sequential(nn.MaxPool2d(2), ConvBnLeakyReLU(128, 256))
        self.stage6 = nn.Sequential(nn.MaxPool2d(2), 
                                    ConvBnLeakyReLU(256, 512),
                                    #nn.ZeroPad2d((0, 1, 0, 1)),
                                    #nn.MaxPool2d((2, 2), 1)
                                    ConvBnLeakyReLU(512, 1024))
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def extract_featrues(self, x):
        feature_pyramid = {}

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.stage5(x)
        feature_pyramid["stride 16"] = x

        x = self.stage6(x)
        feature_pyramid["stride 32"] = x
        
        return feature_pyramid

    def forward(self, x):
        feature_pyramid = self.extract_featrues(x)

        x = self.gap(feature_pyramid["stride 32"])
        x = x.flatten(start_dim=2)
        x = self.fc(x)
        
        return x



class DarkNetTiny(nn.Module):
    def __init__(self):
        super(DarkNetTiny, self).__init__()

        self.conv_1 = Conv_BN_LeakyReLU(3, 16, 3, 1)
        self.maxpool_1 = nn.MaxPool2d((2, 2), 2)  # stride = 2

        self.conv_2 = Conv_BN_LeakyReLU(16, 32, 3, 1)
        self.maxpool_2 = nn.MaxPool2d((2, 2), 2)  # stride = 4

        self.conv_3 = Conv_BN_LeakyReLU(32, 64, 3, 1)
        self.maxpool_3 = nn.MaxPool2d((2, 2), 2)  # stride = 8

        self.conv_4 = Conv_BN_LeakyReLU(64, 128, 3, 1)
        self.maxpool_4 = nn.MaxPool2d((2, 2), 2)  # stride = 16

        self.conv_5 = Conv_BN_LeakyReLU(128, 256, 3, 1)
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)  # stride = 32

        self.conv_6 = Conv_BN_LeakyReLU(256, 512, 3, 1)
        self.maxpool_6 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d((2, 2), 1)  # stride = 32
        )
        self.conv_7 = Conv_BN_LeakyReLU(512, 1024, 3, 1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.maxpool_1(x)
        x = self.conv_2(x)
        x = self.maxpool_2(x)
        x = self.conv_3(x)
        x = self.maxpool_3(x)
        x = self.conv_4(x)
        x = self.maxpool_4(x)
        C_4 = self.conv_5(x)  # stride = 16
        x = self.maxpool_5(C_4)
        x = self.conv_6(x)
        x = self.maxpool_6(x)
        C_5 = self.conv_7(x)  # stride = 32

        return C_4, C_5

def darknet_tiny(weight_path=None):
    model = DarkNetTiny()
    if weight_path:
        model.load_state_dict(torch.load(weight_path, map_location='cpu'),
                              strict=False)
    return model

if __name__ == '__main__':
    model_pretrained = darknet_tiny('backbone_weights/darknet_light_90_58.99.pth')
    model_scratch = darknet_tiny()

    rand_tensor = torch.randn((1, 3, 416, 416))
    print(model_pretrained(rand_tensor)[1].sum())
    print(model_scratch(rand_tensor)[1].sum())
