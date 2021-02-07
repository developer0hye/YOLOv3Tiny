import torch.nn as nn

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
                                    nn.ZeroPad2d((0, 1, 0, 1)),
                                    nn.MaxPool2d((2, 2), 1),
                                    ConvBnLeakyReLU(512, 1024))

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        
        return x