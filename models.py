import torch.nn as nn
import torch.autograd.profiler as profiler

class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2  = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 =  nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):

        with profiler.record_function("CONV-BLOCK1"):
            upsample1 = self.max_pool1(self.relu1(self.bn1(self.conv1(x))))
        with profiler.record_function("CONV-BLOCK2"):
            upsample2 = self.relu2(self.bn2(self.conv2(upsample1)))
        with profiler.record_function("CONV-BLOCK3"):
            upsample3 = self.relu3(self.bn3(self.conv3(upsample2)))
        with profiler.record_function("CONV-BLOCK4"):
            upsample4 = self.relu4(self.bn4(self.conv4(upsample3)))
        with profiler.record_function("POOLING(AVERAGE)"):
            x = self.avgpool(upsample4)
        with profiler.record_function("CLASSIFIER (LINEAR)"):
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        return x

    def forward_with_profiler(self, x, profiler):

        profiler.start()

        upsample1 = self.max_pool1(self.relu1(self.bn1(self.conv1(x))))
        upsample2 = self.relu2(self.bn2(self.conv2(upsample1)))
        upsample3 = self.relu3(self.bn3(self.conv3(upsample2)))
        upsample4 = self.relu4(self.bn4(self.conv4(upsample3)))
        x = self.avgpool(upsample4)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        profiler.stop()

        return x
    

class RNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super(RNetModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
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
        x = self.fc(x)

        return x

    def forward_with_profiler(self, x, profiler):
        profiler.start()

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
        x = self.fc(x)

        profiler.stop()

        return x
    

class MNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MNetModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            self._make_dw_conv_block(32, 64, 1),
            self._make_dw_conv_block(64, 128, 2),
            self._make_dw_conv_block(128, 256, 2),
            self._make_dw_conv_block(256, 512, 2),
            self._make_dw_conv_block(512, 1024, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, num_classes)

    def _make_dw_conv_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def forward_with_profiler(self, x, profiler):
        
        profiler.start()

        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        profiler.stop()

        return x