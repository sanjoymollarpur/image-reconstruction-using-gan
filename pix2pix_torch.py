import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Define layers
        self.conv11 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.conv22 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv23 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv32 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv33 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv42 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv43 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv52 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv45 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv46 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv47 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv35 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv36 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv37 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv39 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv310 = nn.Conv2d(384, 128, kernel_size=3, padding=1)
        self.conv311 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv25 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv26 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv27 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv210 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv211 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.conv212 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv215 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv216 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.conv217 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv14 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv15 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv16 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.conv18 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv19 = nn.Conv2d(96, 32, kernel_size=3, padding=1)
        self.conv110 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.conv112 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv113 = nn.Conv2d(96, 32, kernel_size=3, padding=1)
        self.conv114 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.conv118 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv119 = nn.Conv2d(224, 32, kernel_size=3, padding=1)
        self.conv120 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.output_conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.output_conv2 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, input_tensor):
        conv11 = F.relu(self.conv11(input_tensor))
        conv11 = F.relu(self.conv12(conv11))

        conv12 = F.relu(self.conv12(conv11))

        conv13 = conv12

        maxpool21 = F.max_pool2d(conv12, kernel_size=2, stride=2)

        conv22 = F.relu(self.conv22(maxpool21))
        conv22 = F.relu(self.conv23(conv22))

        conv23 = F.relu(self.conv23(conv22))

        maxpool31 = F.max_pool2d(conv23, kernel_size=2, stride=2)

        conv32 = F.relu(self.conv32(maxpool31))
        conv32 = F.relu(self.conv33(conv32))

        conv33 = F.relu(self.conv33(conv32))

        maxpool41 = F.max_pool2d(conv33, kernel_size=2, stride=2)

        conv42 = F.relu(self.conv42(maxpool41))
        conv42 = F.relu(self.conv43(conv42))

        conv43 = F.relu(self.conv43(conv42))

        maxpool51 = F.max_pool2d(conv43, kernel_size=2, stride=2)

        conv52 = F.relu(self.conv52(maxpool51))
        conv52 = F.relu(self.conv53(conv52))

        conv53 = F.relu(self.conv53(conv52))

        conv44 = conv43

        conv45 = F.relu(self.conv45(conv53))
        conv45 = torch.cat((conv44, conv45), dim=1)

        conv46 = F.relu(self.conv46(conv45))
        conv46 = F.relu(self.conv47(conv46))

        conv47 = F.relu(self.conv47(conv46))

        conv34 = conv33

        conv35 = F.relu(self.conv35(conv43))
        conv35 = torch.cat((conv34, conv35), dim=1)

        conv36 = F.relu(self.conv36(conv35))
        conv36 = F.relu(self.conv37(conv36))

        conv37 = conv33

        conv38 = conv36

        conv39 = F.relu(self.conv39(conv47))
        conv39 = torch.cat((conv37, conv38, conv39), dim=1)

        conv310 = F.relu(self.conv310(conv39))
        conv310 = F.relu(self.conv311(conv310))

        conv311 = F.relu(self.conv311(conv310))

        conv24 = conv23

        conv25 = F.relu(self.conv25(conv33))
        conv25 = torch.cat((conv24, conv25), dim=1)

        conv26 = F.relu(self.conv26(conv25))
        conv26 = F.relu(self.conv27(conv26))

        conv27 = conv23

        conv27 = torch.cat((conv26, conv27), dim=1)

        conv28 = conv27

        conv29 = conv23

        conv210 = F.relu(self.conv210(conv36))
        conv210 = torch.cat((conv28, conv29, conv210), dim=1)

        conv211 = F.relu(self.conv211(conv210))
        conv211 = F.relu(self.conv212(conv211))

        conv212 = conv211

        conv213 = conv23

        conv214 = conv26

        conv215 = F.relu(self.conv215(conv311))
        conv215 = torch.cat((conv212, conv213, conv214, conv215), dim=1)

        conv216 = F.relu(self.conv216(conv215))
        conv216 = F.relu(self.conv217(conv216))

        conv217 = F.relu(self.conv217(conv216))

        conv14 = F.relu(self.conv14(conv23))
        conv14 = torch.cat((conv13, conv14), dim=1)

        conv15 = F.relu(self.conv15(conv14))
        conv15 = F.relu(self.conv16(conv15))

        conv16 = conv15
        conv17 = conv12

        conv18 = F.relu(self.conv18(conv26))
        conv18 = torch.cat((conv16, conv17, conv18), dim=1)

        conv19 = F.relu(self.conv19(conv18))
        conv19 = F.relu(self.conv110(conv19))

        conv110 = conv19

        conv111 = conv15

        conv112 = conv12

        conv112 = torch.cat((conv110, conv111, conv112), dim=1)

        conv113 = F.relu(self.conv113(conv112))
        conv113 = F.relu(self.conv114(conv113))

        conv114 = conv113

        conv115 = conv19

        conv116 = conv115

        conv117 = conv112

        conv118 = F.relu(self.conv118(conv217))
        conv118 = torch.cat((conv114, conv115, conv116, conv117, conv118), dim=1)

        conv119 = F.relu(self.conv119(conv118))
        conv119 = F.relu(self.conv120(conv119))

        conv120 = F.relu(self.output_conv1(conv119))
        output = F.relu(self.output_conv2(conv120))

        return output

# # Example input tensors (batch_size, channels, height, width)
inp = torch.randn(1, 1, 512, 512)  # Replace with your actual input tensor
tar = torch.randn(1, 1, 512, 512)  # Replace with your actual target tensor
# model=Generator()
# output = model(inp)
# print(output.shape)




class BasicBlock(nn.Module):
    """Basic block"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, norm=True):
        super().__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding)
        self.isn = None
        if norm:
            self.isn = nn.InstanceNorm2d(outplanes)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        fx = self.conv(x)
        
        if self.isn is not None:
            fx = self.isn(fx)
            
        fx = self.lrelu(fx)
        return fx
    
class Discriminator(nn.Module):
    """Conditional Discriminator"""
    def __init__(self,):
        super().__init__()
        self.block1 = BasicBlock(2, 64, norm=False)
        self.block2 = BasicBlock(64, 128)
        self.block3 = BasicBlock(128, 256)
        self.block4 = BasicBlock(256, 512)
        self.block5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        
    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        # blocks forward
        fx = self.block1(x)
        fx = self.block2(fx)
        fx = self.block3(fx)
        fx = self.block4(fx)
        fx = self.block5(fx)
        return fx

model=Discriminator()
output = model(inp,tar)
print(output.shape)

# def downsample(x, in_channels, out_channels, kernel_size, apply_batchnorm=True):
#     x = F.conv2d(x, torch.nn.init.normal_(torch.empty(out_channels, in_channels, kernel_size, kernel_size), 0.0, 0.02), stride=2, padding=1)
#     if apply_batchnorm:
#         x = F.batch_norm(x, torch.zeros(out_channels), torch.ones(out_channels))
#     x = F.leaky_relu(x, 0.2)
#     return x

# def discriminator(inp, tar):
#     x = torch.cat([inp, tar], dim=1) 
#     x = downsample(x, x.shape[1], 64, 4, apply_batchnorm=False)  
#     x = downsample(x, 64, 128, 4)
#     x = downsample(x, 128, 256, 4)
#     x = F.pad(x, (1, 1, 1, 1))  
#     x = F.conv2d(x, torch.nn.init.normal_(torch.empty(512, 256, 4, 4), 0.0, 0.02), stride=1)
#     x = F.batch_norm(x, torch.zeros(512), torch.ones(512))
#     x = F.leaky_relu(x, 0.2)
#     x = F.pad(x, (1, 1, 1, 1))  
#     x = F.conv2d(x, torch.nn.init.normal_(torch.empty(1, 512, 4, 4), 0.0, 0.02), stride=1)
#     return x


# # Forward pass
# output = discriminator(inp, tar)
# print(output.shape)









# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# device='cuda'
# device = torch.device(device)

# def residual_block(x, num_layer, device):
#     x = F.max_pool2d(x, kernel_size=(2, 2))
#     sh = x.clone()
#     sh = nn.Conv2d(sh.shape[1], num_layer, kernel_size=1, padding=0).to(device)(sh)
#     sh = nn.BatchNorm2d(sh.shape[1]).to(device)(sh)
#     sh = F.relu(sh)
#     conv_1 = nn.Conv2d(x.shape[1], num_layer, kernel_size=3, padding=1).to(device)(x)
#     conv_1 = nn.BatchNorm2d(conv_1.shape[1]).to(device)(conv_1)
#     conv_1 = F.relu(conv_1)
#     conv_1 = nn.Conv2d(conv_1.shape[1], num_layer, kernel_size=3, padding=1).to(device)(conv_1)
#     conv_1 = nn.BatchNorm2d(conv_1.shape[1]).to(device)(conv_1)
#     conv_1 = conv_1 + sh  # Avoid in-place operation
#     conv_1 = F.relu(conv_1)
#     return conv_1

# def bn_relu(channels, samp, device):
#     samp = nn.BatchNorm2d(channels).to(device)(samp)
#     samp = F.relu(samp)
#     return samp

# class Generator(nn.Module):
#     def __init__(self, input_size):
#         super(Generator, self).__init__()
#         self.input_size = input_size
#         self.sh1 = nn.Conv2d(input_size[0], 64, kernel_size=1, padding=0).to(device)
#         self.conv1_1 = nn.Conv2d(input_size[0], 32, kernel_size=3, padding=1).to(device)
#         self.conv1_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1).to(device)
#         self.res_block2 = residual_block
#         self.res_block3 = residual_block
#         self.res_block4 = residual_block
#         self.pool = nn.MaxPool2d(kernel_size=(2, 2))
#         self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1).to(device)
#         self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1).to(device)
#         self.conv7_1 = nn.Conv2d(1024, 256, kernel_size=1).to(device)
#         self.conv7_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1).to(device)
#         self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1).to(device)
#         self.conv8_1 = nn.Conv2d(512, 128, kernel_size=1).to(device)
#         self.conv8_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1).to(device)
#         self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1).to(device)
#         self.conv9_1 = nn.Conv2d(256, 64, kernel_size=1).to(device)
#         self.conv9_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1).to(device)
#         self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1).to(device)
#         self.conv10_1 = nn.Conv2d(128, 32, kernel_size=1).to(device)
#         self.conv10_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1).to(device)
#         self.conv10_3 = nn.Conv2d(64, 1, kernel_size=1).to(device)

#     def forward(self, x):
#         inputs = x
#         sh1 = self.sh1(inputs)
#         sh1 = bn_relu(64, sh1, device)
#         conv1 = self.conv1_1(inputs)
#         conv1 = bn_relu(32, conv1, device)
#         conv1 = self.conv1_2(conv1)
#         conv1 = bn_relu(64, conv1, device)
#         conv1 = conv1 + sh1  # Avoid in-place operation
#         conv2 = self.res_block2(conv1, 128, device)
#         conv3 = self.res_block3(conv2, 256, device)
#         conv4 = self.res_block4(conv3, 512, device)
#         sh2 = conv4.clone()
#         conv4 = self.pool(conv4)
#         conv5 = self.conv5(conv4)
#         conv5 = bn_relu(1024, conv5, device)
#         up1 = self.up1(conv5)
#         up1 = bn_relu(512, up1, device)
#         merge1 = torch.cat((up1, sh2), dim=1)
#         conv7 = self.conv7_1(merge1)
#         conv7 = self.conv7_2(conv7)
#         conv7 = bn_relu(512, conv7, device)
#         up2 = self.up2(conv7)
#         up2 = bn_relu(256, up2, device)
#         merge2 = torch.cat((up2, conv3), dim=1)
#         conv8 = self.conv8_1(merge2)
#         conv8 = self.conv8_2(conv8)
#         conv8 = bn_relu(256, conv8, device)
#         up3 = self.up3(conv8)
#         up3 = bn_relu(128, up3, device)
#         merge3 = torch.cat((up3, conv2), dim=1)
#         conv9 = self.conv9_1(merge3)
#         conv9 = self.conv9_2(conv9)
#         conv9 = bn_relu(128, conv9, device)
#         up4 = self.up4(conv9)
#         up4 = bn_relu(64, up4, device)
#         merge4 = torch.cat((up4, conv1), dim=1)
#         conv10 = self.conv10_1(merge4)
#         conv10 = self.conv10_2(conv10)
#         conv10 = bn_relu(64, conv10, device)
#         conv10 = self.conv10_3(conv10)
#         output = conv10 + inputs
#         return output

# input_size = (1, 512, 512)  # Example input size
# gen_model = Generator(input_size)
# inputs = torch.randn(1, *input_size).to(device)
# outputs = gen_model(inputs)
# print(outputs.shape)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def generator(input_tensor):
#     conv11 = F.relu(nn.Conv2d(1, 32, kernel_size=3, padding=1)(input_tensor))
#     conv11 = F.relu(nn.Conv2d(32, 32, kernel_size=3, padding=1)(conv11))

#     conv12 = F.relu(nn.Conv2d(32, 32, kernel_size=3, padding=1)(conv11))

#     conv13 = conv12

#     maxpool21 = F.max_pool2d(conv12, kernel_size=2, stride=2)

#     conv22 = F.relu(nn.Conv2d(32, 64, kernel_size=3, padding=1)(maxpool21))
#     conv22 = F.relu(nn.Conv2d(64, 64, kernel_size=3, padding=1)(conv22))

#     conv23 = F.relu(nn.Conv2d(64, 64, kernel_size=3, padding=1)(conv22))

#     maxpool31 = F.max_pool2d(conv23, kernel_size=2, stride=2)

#     conv32 = F.relu(nn.Conv2d(64, 128, kernel_size=3, padding=1)(maxpool31))
#     conv32 = F.relu(nn.Conv2d(128, 128, kernel_size=3, padding=1)(conv32))

#     conv33 = F.relu(nn.Conv2d(128, 128, kernel_size=3, padding=1)(conv32))

#     maxpool41 = F.max_pool2d(conv33, kernel_size=2, stride=2)

#     conv42 = F.relu(nn.Conv2d(128, 256, kernel_size=3, padding=1)(maxpool41))
#     conv42 = F.relu(nn.Conv2d(256, 256, kernel_size=3, padding=1)(conv42))

#     conv43 = F.relu(nn.Conv2d(256, 256, kernel_size=3, padding=1)(conv42))

#     maxpool51 = F.max_pool2d(conv43, kernel_size=2, stride=2)

#     conv52 = F.relu(nn.Conv2d(256, 512, kernel_size=3, padding=1)(maxpool51))
#     conv52 = F.relu(nn.Conv2d(512, 512, kernel_size=3, padding=1)(conv52))

#     conv53 = F.relu(nn.Conv2d(512, 512, kernel_size=3, padding=1)(conv52))

#     conv44 = conv43

#     conv45 = F.relu(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)(conv53))
#     conv45 = torch.cat((conv44, conv45), dim=1)

#     conv46 = F.relu(nn.Conv2d(512, 256, kernel_size=3, padding=1)(conv45))
#     conv46 = F.relu(nn.Conv2d(256, 256, kernel_size=3, padding=1)(conv46))

#     conv47 = F.relu(nn.Conv2d(256, 256, kernel_size=3, padding=1)(conv46))

#     conv34 = conv33

#     conv35 = F.relu(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)(conv43))
#     conv35 = torch.cat((conv34, conv35), dim=1)

#     conv36 = F.relu(nn.Conv2d(256, 128, kernel_size=3, padding=1)(conv35))
#     conv36 = F.relu(nn.Conv2d(128, 128, kernel_size=3, padding=1)(conv36))

#     conv37 = conv33

#     conv38 = conv36

#     conv39 = F.relu(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)(conv47))
#     conv39 = torch.cat((conv37, conv38, conv39), dim=1)

#     conv310 = F.relu(nn.Conv2d(384, 128, kernel_size=3, padding=1)(conv39))
#     conv310 = F.relu(nn.Conv2d(128, 128, kernel_size=3, padding=1)(conv310))

#     conv311 = F.relu(nn.Conv2d(128, 128, kernel_size=3, padding=1)(conv310))

#     conv24 = conv23

#     conv25 = F.relu(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)(conv33))
#     conv25 = torch.cat((conv24, conv25), dim=1)

#     conv26 = F.relu(nn.Conv2d(128, 64, kernel_size=3, padding=1)(conv25))
#     conv26 = F.relu(nn.Conv2d(64, 64, kernel_size=3, padding=1)(conv26))

#     conv27 = conv23

#     conv27 = torch.cat((conv26, conv27), dim=1)

#     conv28 = conv27

#     conv29 = conv23

#     conv210 = F.relu(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)(conv36))
#     conv210 = torch.cat((conv28, conv29, conv210), dim=1)

#     conv211 = F.relu(nn.Conv2d(256, 64, kernel_size=3, padding=1)(conv210))
#     conv211 = F.relu(nn.Conv2d(64, 64, kernel_size=3, padding=1)(conv211))

#     conv212 = conv211

#     conv213 = conv23

#     conv214 = conv26

#     conv215 = F.relu(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)(conv311))
#     conv215 = torch.cat((conv212, conv213, conv214, conv215), dim=1)

#     conv216 = F.relu(nn.Conv2d(256, 64, kernel_size=3, padding=1)(conv215))
#     conv216 = F.relu(nn.Conv2d(64, 64, kernel_size=3, padding=1)(conv216))

#     conv217 = F.relu(nn.Conv2d(64, 64, kernel_size=3, padding=1)(conv216))

#     conv14 = F.relu(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)(conv23))
#     conv14 = torch.cat((conv13, conv14), dim=1)

#     conv15 = F.relu(nn.Conv2d(64, 32, kernel_size=3, padding=1)(conv14))
#     conv15 = F.relu(nn.Conv2d(32, 32, kernel_size=3, padding=1)(conv15))

#     conv16 = conv15

#     conv17 = conv12

#     conv18 = F.relu(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)(conv26))
#     conv18 = torch.cat((conv16, conv17, conv18), dim=1)

#     conv19 = F.relu(nn.Conv2d(96, 32, kernel_size=3, padding=1)(conv18))
#     conv19 = F.relu(nn.Conv2d(32, 32, kernel_size=3, padding=1)(conv19))

#     conv110 = conv19

#     conv111 = conv15

#     conv112 = conv12

#     conv112 = torch.cat((conv110, conv111, conv112), dim=1)

#     conv113 = F.relu(nn.Conv2d(96, 32, kernel_size=3, padding=1)(conv112))
#     conv113 = F.relu(nn.Conv2d(32, 32, kernel_size=3, padding=1)(conv113))

#     conv114 = conv113

#     conv115 = conv19

#     conv116 = conv115

#     conv117 = conv112

#     conv118 = F.relu(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)(conv217))
#     conv118 = torch.cat((conv114, conv115, conv116, conv117, conv118), dim=1)

#     conv119 = F.relu(nn.Conv2d(224, 32, kernel_size=3, padding=1)(conv118))
#     conv119 = F.relu(nn.Conv2d(32, 32, kernel_size=3, padding=1)(conv119))

#     conv120 = F.relu(nn.Conv2d(32, 32, kernel_size=3, padding=1)(conv119))

#     output = F.relu(nn.Conv2d(32, 32, kernel_size=3, padding=1)(conv120))
#     output = F.relu(nn.Conv2d(32, 1, kernel_size=3, padding=1)(output))

#     return output




# class Discriminator(nn.Module):
#     def __init__(self, input_size):
#         super(Discriminator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(input_size[1], 64, kernel_size=3, padding=1).to(device),  # padding='same'
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1).to(device),  # strides=2, padding='same'
#             nn.BatchNorm2d(64).to(device),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(64, 128, kernel_size=3, padding=1).to(device),  # padding='same'
#             nn.BatchNorm2d(128).to(device),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1).to(device),  # strides=2, padding='same'
#             nn.BatchNorm2d(128).to(device),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(128, 256, kernel_size=3, padding=1).to(device),  # padding='same'
#             nn.BatchNorm2d(256).to(device),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1).to(device),  # strides=2, padding='same'
#             nn.BatchNorm2d(256).to(device),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(256, 512, kernel_size=3, padding=1).to(device),  # padding='same'
#             nn.BatchNorm2d(512).to(device),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1).to(device),  # strides=2, padding='same'
#             nn.BatchNorm2d(512).to(device),
#             nn.LeakyReLU(0.2),

#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(512, 1024, kernel_size=1).to(device),  # Dense(1024)
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(1024, 1, kernel_size=1).to(device)  # Dense(1)
#         )

#     def forward(self, x):
#         return self.model(x)

# input_tensor = torch.randn(1, 1, 512, 512).to(device)  # Example input tensor with shape (batch_size, channels, height, width)
# disc_model = Discriminator(input_tensor.shape)
# output = disc_model(input_tensor)
# print(output.shape)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def downsample(x, in_channels, out_channels, kernel_size, apply_batchnorm=True):
#     x = F.conv2d(x, torch.nn.init.normal_(torch.empty(out_channels, in_channels, kernel_size, kernel_size), 0.0, 0.02), stride=2, padding=1)
#     if apply_batchnorm:
#         x = F.batch_norm(x, torch.zeros(out_channels), torch.ones(out_channels))
#     x = F.leaky_relu(x, 0.2)
#     return x
# def discriminator(inp, tar, what):
#     if what:
#         x = torch.cat([inp, tar], dim=1) 
#     else:
#         x=inp 
    
#     x = downsample(x, x.shape[1], 64, 4, apply_batchnorm=False)  
#     x = downsample(x, 64, 128, 4)
#     x = downsample(x, 128, 256, 4)
    
#     x = F.pad(x, (1, 1, 1, 1))  
#     x = F.conv2d(x, torch.nn.init.normal_(torch.empty(512, 256, 4, 4), 0.0, 0.02), stride=1)
#     x = F.batch_norm(x, torch.zeros(512), torch.ones(512))
#     x = F.leaky_relu(x, 0.2)
    
#     x = F.pad(x, (1, 1, 1, 1))  
#     x = F.conv2d(x, torch.nn.init.normal_(torch.empty(1, 512, 4, 4), 0.0, 0.02), stride=1)
    
#     return x

# # Example input tensors (batch_size, channels, height, width)
# inp = torch.randn(1, 1, 512, 512)  # Replace with your actual input tensor
# tar = torch.randn(1, 1, 512, 512)  # Replace with your actual target tensor

# # Forward pass
# output = discriminator(inp, tar,False)
# print(output.shape)


# import torch
# import torch.nn.functional as F
# import torch.nn as nn

# def generator(input_tensor):
#     layers = []

#     conv11 = F.relu(nn.Conv2d(1, 32, kernel_size=3, padding=1)(input_tensor))
#     layers.append(conv11)
#     conv11 = F.relu(nn.Conv2d(32, 32, kernel_size=3, padding=1)(conv11))
#     layers.append(conv11)

#     conv12 = F.relu(nn.Conv2d(32, 32, kernel_size=3, padding=1)(conv11))
#     layers.append(conv12)

#     conv13 = conv12

#     maxpool21 = F.max_pool2d(conv12, kernel_size=2, stride=2)
#     layers.append(maxpool21)

#     conv22 = F.relu(nn.Conv2d(32, 64, kernel_size=3, padding=1)(maxpool21))
#     layers.append(conv22)
#     conv22 = F.relu(nn.Conv2d(64, 64, kernel_size=3, padding=1)(conv22))
#     layers.append(conv22)

#     conv23 = F.relu(nn.Conv2d(64, 64, kernel_size=3, padding=1)(conv22))
#     layers.append(conv23)

#     maxpool31 = F.max_pool2d(conv23, kernel_size=2, stride=2)
#     layers.append(maxpool31)

#     conv32 = F.relu(nn.Conv2d(64, 128, kernel_size=3, padding=1)(maxpool31))
#     layers.append(conv32)
#     conv32 = F.relu(nn.Conv2d(128, 128, kernel_size=3, padding=1)(conv32))
#     layers.append(conv32)

#     conv33 = F.relu(nn.Conv2d(128, 128, kernel_size=3, padding=1)(conv32))
#     layers.append(conv33)

#     maxpool41 = F.max_pool2d(conv33, kernel_size=2, stride=2)
#     layers.append(maxpool41)

#     conv42 = F.relu(nn.Conv2d(128, 256, kernel_size=3, padding=1)(maxpool41))
#     layers.append(conv42)
#     conv42 = F.relu(nn.Conv2d(256, 256, kernel_size=3, padding=1)(conv42))
#     layers.append(conv42)

#     conv43 = F.relu(nn.Conv2d(256, 256, kernel_size=3, padding=1)(conv42))
#     layers.append(conv43)

#     maxpool51 = F.max_pool2d(conv43, kernel_size=2, stride=2)
#     layers.append(maxpool51)

#     conv52 = F.relu(nn.Conv2d(256, 512, kernel_size=3, padding=1)(maxpool51))
#     layers.append(conv52)
#     conv52 = F.relu(nn.Conv2d(512, 512, kernel_size=3, padding=1)(conv52))
#     layers.append(conv52)

#     conv53 = F.relu(nn.Conv2d(512, 512, kernel_size=3, padding=1)(conv52))
#     layers.append(conv53)

#     conv44 = conv43

#     conv45 = F.relu(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)(conv53))
#     layers.append(conv45)
#     conv45 = torch.cat((conv44, conv45), dim=1)
#     layers.append(conv45)

#     conv46 = F.relu(nn.Conv2d(512, 256, kernel_size=3, padding=1)(conv45))
#     layers.append(conv46)
#     conv46 = F.relu(nn.Conv2d(256, 256, kernel_size=3, padding=1)(conv46))
#     layers.append(conv46)

#     conv47 = F.relu(nn.Conv2d(256, 256, kernel_size=3, padding=1)(conv46))
#     layers.append(conv47)

#     conv34 = conv33

#     conv35 = F.relu(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)(conv43))
#     layers.append(conv35)
#     conv35 = torch.cat((conv34, conv35), dim=1)
#     layers.append(conv35)

#     conv36 = F.relu(nn.Conv2d(256, 128, kernel_size=3, padding=1)(conv35))
#     layers.append(conv36)
#     conv36 = F.relu(nn.Conv2d(128, 128, kernel_size=3, padding=1)(conv36))
#     layers.append(conv36)

#     conv37 = conv33
#     conv38 = conv36

#     conv39 = F.relu(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)(conv47))
#     layers.append(conv39)
#     conv39 = torch.cat((conv37, conv38, conv39), dim=1)
#     layers.append(conv39)

#     conv310 = F.relu(nn.Conv2d(384, 128, kernel_size=3, padding=1)(conv39))
#     layers.append(conv310)
#     conv310 = F.relu(nn.Conv2d(128, 128, kernel_size=3, padding=1)(conv310))
#     layers.append(conv310)

#     conv311 = F.relu(nn.Conv2d(128, 128, kernel_size=3, padding=1)(conv310))
#     layers.append(conv311)

#     conv24 = conv23

#     conv25 = F.relu(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)(conv33))
#     layers.append(conv25)
#     conv25 = torch.cat((conv24, conv25), dim=1)
#     layers.append(conv25)

#     conv26 = F.relu(nn.Conv2d(128, 64, kernel_size=3, padding=1)(conv25))
#     layers.append(conv26)
#     conv26 = F.relu(nn.Conv2d(64, 64, kernel_size=3, padding=1)(conv26))
#     layers.append(conv26)

#     conv27 = conv23
#     conv27 = torch.cat((conv26, conv27), dim=1)
#     layers.append(conv27)

#     conv28 = conv27
#     conv29 = conv23

#     conv210 = F.relu(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)(conv36))
#     layers.append(conv210)
#     conv210 = torch.cat((conv28, conv29, conv210), dim=1)
#     layers.append(conv210)

#     conv211 = F.relu(nn.Conv2d(256, 64, kernel_size=3, padding=1)(conv210))
#     layers.append(conv211)
#     conv211 = F.relu(nn.Conv2d(64, 64, kernel_size=3, padding=1)(conv211))
#     layers.append(conv211)

#     conv212 = conv211
#     conv213 = conv23
#     conv214 = conv26

#     conv215 = F.relu(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)(conv311))
#     layers.append(conv215)
#     conv215 = torch.cat((conv212, conv213, conv214, conv215), dim=1)
#     layers.append(conv215)

#     conv216 = F.relu(nn.Conv2d(256, 64, kernel_size=3, padding=1)(conv215))
#     layers.append(conv216)
#     conv216 = F.relu(nn.Conv2d(64, 64, kernel_size=3, padding=1)(conv216))
#     layers.append(conv216)

#     conv217 = F.relu(nn.Conv2d(64, 64, kernel_size=3, padding=1)(conv216))
#     layers.append(conv217)

#     conv14 = F.relu(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)(conv23))
#     layers.append(conv14)
#     conv14 = torch.cat((conv13, conv14), dim=1)
#     layers.append(conv14)

#     conv15 = F.relu(nn.Conv2d(64, 32, kernel_size=3, padding=1)(conv14))
#     layers.append(conv15)
#     conv15 = F.relu(nn.Conv2d(32, 32, kernel_size=3, padding=1)(conv15))
#     layers.append(conv15)

#     conv16 = conv15
#     conv17 = conv12

#     conv18 = F.relu(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)(conv26))
#     layers.append(conv18)
#     conv18 = torch.cat((conv16, conv17, conv18), dim=1)
#     layers.append(conv18)

#     conv19 = F.relu(nn.Conv2d(96, 32, kernel_size=3, padding=1)(conv18))
#     layers.append(conv19)
#     conv19 = F.relu(nn.Conv2d(32, 32, kernel_size=3, padding=1)(conv19))
#     layers.append(conv19)

#     conv110 = conv19
#     conv111 = conv15
#     conv112 = conv12
#     conv112 = torch.cat((conv110, conv111, conv112), dim=1)
#     layers.append(conv112)

#     conv113 = F.relu(nn.Conv2d(96, 32, kernel_size=3, padding=1)(conv112))
#     layers.append(conv113)
#     conv113 = F.relu(nn.Conv2d(32, 32, kernel_size=3, padding=1)(conv113))
#     layers.append(conv113)

#     conv114 = conv113

#     conv115 = conv19
#     conv116 = conv115
#     conv117 = conv112

#     conv118 = F.relu(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)(conv217))
#     layers.append(conv118)
#     conv118 = torch.cat((conv114, conv115, conv116, conv117, conv118), dim=1)
#     layers.append(conv118)

#     conv119 = F.relu(nn.Conv2d(224, 32, kernel_size=3, padding=1)(conv118))
#     layers.append(conv119)
#     conv119 = F.relu(nn.Conv2d(32, 32, kernel_size=3, padding=1)(conv119))
#     layers.append(conv119)

#     conv120 = F.relu(nn.Conv2d(32, 32, kernel_size=3, padding=1)(conv119))
#     layers.append(conv120)

#     output = F.relu(nn.Conv2d(32, 32, kernel_size=3, padding=1)(conv120))
#     layers.append(output)
#     output = F.relu(nn.Conv2d(32, 1, kernel_size=3, padding=1)(output))
#     layers.append(output)

#     model = {
#         'input': input_tensor,
#         'output': output
#     }

#     return layers,model

# input_tensor = torch.randn(1, 1, 512, 512)  # Assuming a random input tensor of shape (1, 1, 512, 512)

# # Run the generator function to get the model and output
# layers,model = generator(input_tensor)

# # Access the input and output from the model dictionary
# input_data = model['input']
# output_data = model['output']

# print(output_data.shape)
# import torch.optim as optim

# optimizer = optim.Adam([param for layer in layers for param in layer.parameters() if param.requires_grad], lr=0.001)


# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# import torch.optim as optim

# def generator(input_tensor):
#     layers = []

#     # Define convolutional layers
#     conv11 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#     layers.append(conv11)
#     conv12 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
#     layers.append(conv12)

#     conv21 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#     layers.append(conv21)
#     conv22 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#     layers.append(conv22)

#     conv31 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#     layers.append(conv31)
#     conv32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#     layers.append(conv32)

#     conv41 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#     layers.append(conv41)
#     conv42 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#     layers.append(conv42)

#     conv51 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#     layers.append(conv51)
#     conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#     layers.append(conv52)

#     # Define transposed convolutional layers
#     conv45 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
#     layers.append(conv45)
#     conv35 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
#     layers.append(conv35)
#     conv39 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
#     layers.append(conv39)
#     conv25 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
#     layers.append(conv25)
#     conv18 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
#     layers.append(conv18)
#     conv14 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
#     layers.append(conv14)

#     # Define additional convolutional layers
#     conv15 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
#     layers.append(conv15)
#     conv16 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
#     layers.append(conv16)
#     conv19 = nn.Conv2d(96, 32, kernel_size=3, padding=1)
#     layers.append(conv19)
#     conv110 = nn.Conv2d(96, 32, kernel_size=3, padding=1)
#     layers.append(conv110)
#     conv113 = nn.Conv2d(96, 32, kernel_size=3, padding=1)
#     layers.append(conv113)
#     conv115 = nn.Conv2d(224, 32, kernel_size=3, padding=1)
#     layers.append(conv115)
#     conv119 = nn.Conv2d(224, 32, kernel_size=3, padding=1)
#     layers.append(conv119)
#     conv120 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
#     layers.append(conv120)
#     conv121 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
#     layers.append(conv121)

#     # Apply layers to input_tensor
#     x = F.relu(conv11(input_tensor))
#     x = F.relu(conv12(x))

#     x = F.relu(conv21(x))
#     x = F.relu(conv22(x))

#     x = F.relu(conv31(x))
#     x = F.relu(conv32(x))

#     x = F.relu(conv41(x))
#     x = F.relu(conv42(x))

#     x = F.relu(conv51(x))
#     x = F.relu(conv52(x))

#     x = F.relu(conv45(x))
#     x = F.relu(conv35(x))
#     x = F.relu(conv39(x))
#     x = F.relu(conv25(x))
#     x = F.relu(conv18(x))
#     x = F.relu(conv14(x))

#     x = F.relu(conv15(x))
#     x = F.relu(conv16(x))
#     x = F.relu(conv19(x))
#     x = F.relu(conv110(x))
#     x = F.relu(conv113(x))
#     x = F.relu(conv115(x))
#     x = F.relu(conv119(x))
#     x = F.relu(conv120(x))
#     x = F.relu(conv121(x))

#     # Define output layer
#     output = F.relu(nn.Conv2d(32, 32, kernel_size=3, padding=1)(x))
#     layers.append(output)
#     output = F.relu(nn.Conv2d(32, 1, kernel_size=3, padding=1)(output))
#     layers.append(output)

#     return layers, output

# # Example usage:

# input_tensor = torch.randn(1, 1, 512, 512)  # Example input tensor
# layers, output = generator(input_tensor)
# print(output.shape)

# # Create optimizer
# optimizer = optim.Adam([param for layer in layers for param in layer.parameters() if param.requires_grad], lr=0.001)

# # Example of accessing and printing the parameters of the optimizer
# for param_group in optimizer.param_groups:
#     print(param_group['params'])  # This will print the list of parameters in each param_group
