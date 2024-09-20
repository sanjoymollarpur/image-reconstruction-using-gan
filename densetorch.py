import torch
import torch.nn as nn
import torch.nn.functional as F

class BNReLU(nn.Module):
    def __init__(self, num_features):
        super(BNReLU, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(x))

class DenseBlock(nn.Module):
    def __init__(self, in_channels, pool):
        super(DenseBlock, self).__init__()
        self.pool = pool
        inter_channels = in_channels // 4

        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, padding='same')
        self.bn_relu1 = BNReLU(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding='same')
        self.bn_relu2 = BNReLU(inter_channels)

        self.conv3 = nn.Conv2d(in_channels + inter_channels, inter_channels, kernel_size=1, padding='same')
        self.bn_relu3 = BNReLU(inter_channels)
        self.conv4 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding='same')
        self.bn_relu4 = BNReLU(inter_channels)

        self.conv5 = nn.Conv2d(in_channels + 2 * inter_channels, inter_channels, kernel_size=1, padding='same')
        self.bn_relu5 = BNReLU(inter_channels)
        self.conv6 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding='same')
        self.bn_relu6 = BNReLU(inter_channels)

        self.conv7 = nn.Conv2d(in_channels + 3 * inter_channels, inter_channels, kernel_size=1, padding='same')
        self.bn_relu7 = BNReLU(inter_channels)
        self.conv8 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding='same')
        self.bn_relu8 = BNReLU(inter_channels)

        self.out_channels = in_channels + 4 * inter_channels

    def forward(self, x):
        if self.pool:
            x = F.max_pool2d(x, 2)
        
        conv1 = self.bn_relu1(self.conv1(x))
        conv1 = self.bn_relu2(self.conv2(conv1))
        
        conv2 = torch.cat([x, conv1], dim=1)
        conv2 = self.bn_relu3(self.conv3(conv2))
        conv2 = self.bn_relu4(self.conv4(conv2))
        
        conv3 = torch.cat([x, conv1, conv2], dim=1)
        conv3 = self.bn_relu5(self.conv5(conv3))
        conv3 = self.bn_relu6(self.conv6(conv3))
        
        conv4 = torch.cat([x, conv1, conv2, conv3], dim=1)
        conv4 = self.bn_relu7(self.conv7(conv4))
        conv4 = self.bn_relu8(self.conv8(conv4))
        
        conv5 = torch.cat([x, conv1, conv2, conv3, conv4], dim=1)
        return conv5

class Gen(nn.Module):
    def __init__(self, input_size):
        super(Gen, self).__init__()
        self.conv1 = nn.Conv2d(input_size[0], 64, kernel_size=1, padding='same')
        self.bn_relu1 = BNReLU(64)
        
        self.conv2 = nn.Conv2d(input_size[0], 32, kernel_size=3, padding='same')
        self.bn_relu2 = BNReLU(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.bn_relu3 = BNReLU(64)
        
        self.dense_block1 = DenseBlock(64, False)
        self.dense_block2 = DenseBlock(self.dense_block1.out_channels, True)
        self.dense_block3 = DenseBlock(self.dense_block2.out_channels, True)
        self.dense_block4 = DenseBlock(self.dense_block3.out_channels, True)

        self.maxpool = nn.MaxPool2d(2)

        self.dense_block5 = DenseBlock(self.dense_block4.out_channels, False)

        self.deconv1 = nn.ConvTranspose2d(self.dense_block5.out_channels, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_relu4 = BNReLU(512)
        self.conv4 = nn.Conv2d(512 + self.dense_block4.out_channels, 256, kernel_size=1, padding='same')
        self.dense_block6 = DenseBlock(256, False)

        self.deconv2 = nn.ConvTranspose2d(self.dense_block6.out_channels, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_relu5 = BNReLU(256)
        self.conv5 = nn.Conv2d(256 + self.dense_block3.out_channels, 128, kernel_size=1, padding='same')
        self.dense_block7 = DenseBlock(128, False)

        self.deconv3 = nn.ConvTranspose2d(self.dense_block7.out_channels, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_relu6 = BNReLU(128)
        self.conv6 = nn.Conv2d(128 + self.dense_block2.out_channels, 64, kernel_size=1, padding='same')
        self.dense_block8 = DenseBlock(64, False)

        self.deconv4 = nn.ConvTranspose2d(self.dense_block8.out_channels, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_relu7 = BNReLU(64)
        self.conv7 = nn.Conv2d(64 + self.dense_block1.out_channels, 32, kernel_size=1, padding='same')
        self.dense_block9 = DenseBlock(32, False)

        self.final_conv = nn.Conv2d(self.dense_block9.out_channels, 1, kernel_size=1, padding='same')

    def forward(self, x):
        sh1 = self.bn_relu1(self.conv1(x))
        conv1 = self.bn_relu2(self.conv2(x))
        conv1 = self.bn_relu3(self.conv3(conv1))
        conv1 = conv1 + sh1
        conv1 = self.dense_block1(conv1)

        conv2 = self.dense_block2(conv1)
        conv3 = self.dense_block3(conv2)
        conv4 = self.dense_block4(conv3)
        sh2 = conv4
        conv4 = self.maxpool(conv4)

        conv5 = self.dense_block5(conv4)

        up1 = self.bn_relu4(self.deconv1(conv5))
        merge1 = torch.cat([up1, sh2], dim=1)
        conv7 = self.conv4(merge1)
        conv7 = self.dense_block6(conv7)

        up2 = self.bn_relu5(self.deconv2(conv7))
        merge2 = torch.cat([up2, conv3], dim=1)
        conv8 = self.conv5(merge2)
        conv8 = self.dense_block7(conv8)

        up3 = self.bn_relu6(self.deconv3(conv8))
        merge3 = torch.cat([up3, conv2], dim=1)
        conv9 = self.conv6(merge3)
        conv9 = self.dense_block8(conv9)

        up4 = self.bn_relu7(self.deconv4(conv9))
        merge4 = torch.cat([up4, conv1], dim=1)
        conv10 = self.conv7(merge4)
        conv10 = self.dense_block9(conv10)

        output = self.final_conv(conv10) + x
        return output

# input_size = (1, 1,512, 512)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Gen(input_size).to(device)
# dummy_input = torch.randn(1, 1, 512, 512).to(device)
# output = model(dummy_input)
# print(output.shape)
