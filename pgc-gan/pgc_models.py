import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        # [c,h, w] -> [c, h, w]
        conv_block = [nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, ngf=64):
        super(Generator, self).__init__()
        # the number of res blocks
        self.nBlocks = n_residual_blocks

        # for fade in and fade out
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # create from_rgb layer
        mult = 1
        from_rgb_ngf1 = [ nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, int(ngf*mult), kernel_size=7, padding=0, bias=True),
                    nn.InstanceNorm2d(int(ngf*mult), affine=True),
                    nn.ReLU(True)]
        mult = 2
        from_rgb_ngf2 = [nn.ReflectionPad2d(3),
                        nn.Conv2d(input_nc, ngf*mult, kernel_size=7, padding=0, bias=True),
                        nn.InstanceNorm2d(int(ngf*mult), affine=True),
                        nn.ReLU(True)]

        mult = 4
        from_rgb_ngf4 = [nn.ReflectionPad2d(3),
                        nn.Conv2d(input_nc, ngf*mult, kernel_size=7, padding=0, bias=True),
                        nn.InstanceNorm2d(int(ngf*mult), affine=True),
                        nn.ReLU(True)]


        l_from_rgb_ngf4 = nn.Sequential(*from_rgb_ngf4)
        l_from_rgb_ngf2 = nn.Sequential(*from_rgb_ngf2)
        l_from_rgb_ngf1 = nn.Sequential(*from_rgb_ngf1)
        self.fromRGBs = nn.ModuleList([l_from_rgb_ngf4, l_from_rgb_ngf2, l_from_rgb_ngf1])

        # down sampling
        mult = 1
        down_sampling_ngf1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                          nn.InstanceNorm2d(ngf * mult * 2, affine=True),
                          nn.ReLU(True)]
        mult = 2
        down_sampling_ngf2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                          nn.InstanceNorm2d(ngf * mult * 2, affine=True),
                          nn.ReLU(True)]

        l_down_sampling_ngf1 = nn.Sequential(*down_sampling_ngf1)
        l_down_sampling_ngf2 = nn.Sequential(*down_sampling_ngf2)
        self.downSamples = nn.ModuleList([l_down_sampling_ngf2, l_down_sampling_ngf1])

        # add resnet
        mult = 4
        transforms = []
        for i in range(self.nBlocks):  # add ResNet blocks
            transforms += [ResidualBlock(ngf * mult)]
        self.transform = nn.Sequential(*transforms)

        # Up sampling
        up_sampling_ngf1 =[nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3, stride=2, padding=1, output_padding=1),
                           nn.InstanceNorm2d(ngf * mult // 2),
                           nn.ReLU(inplace=True)]
        up_sampling_ngf2 = [nn.ConvTranspose2d(ngf * mult // 2, ngf * mult // 4, 3, stride=2, padding=1, output_padding=1),
                           nn.InstanceNorm2d(ngf * mult // 4),
                           nn.ReLU(inplace=True)]
        l_up_sampling_ngf1 = nn.Sequential(*up_sampling_ngf1)
        l_up_sampling_ngf2 = nn.Sequential(*up_sampling_ngf2)
        self.upSamples = nn.ModuleList([l_up_sampling_ngf1, l_up_sampling_ngf2])

        # to RGB layer
        mult = 4
        to_rgb_1 = [nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf * mult, output_nc, 7, 1),
                    nn.Tanh()]

        to_rgb_2 = [nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf * mult // 2, output_nc, 7, 1),
                    nn.Tanh()]

        to_rgb_4 = [nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf * mult // 4, output_nc, 7, 1),
                    nn.Tanh()]
        l_to_rgb_ngf4 = nn.Sequential(*to_rgb_4)
        l_to_rgb_ngf2 = nn.Sequential(*to_rgb_2)
        l_to_rgb_ngf1 = nn.Sequential(*to_rgb_1)
        self.toRGBs = nn.ModuleList([l_to_rgb_ngf1, l_to_rgb_ngf2, l_to_rgb_ngf4])


    def forward(self, img, step=0, alpha=0):
        # from rgb
        e = self.fromRGBs[step](img)
        old_e = e
        if step != 0:
            old_img = self.downsample(img)
            old_e = self.fromRGBs[step - 1](old_img)
            old_step = step - 1
            for i in range(old_step - 1, -1, -1):
                old_e = self.downSamples[i](old_e)

        # down sample
        for i in range(step - 1, -1, -1):
            e = self.downSamples[i](e)

        # transform
        if step != 0:
            e = (1 - alpha) * old_e + alpha * e
        for i in range(self.nBlocks):
            e = self.transform[i](e)

        # up sample
        old_img = None
        if step != 0:
            old_e = e
            old_step = step - 1
            for i in range(0, old_step):
                old_e = self.upSamples[i](old_e)
            old_img = self.toRGBs[step - 1](old_e)
            old_img = self.upsample(old_img)

        for i in range(0, step):
            e = self.upSamples[i](e)
        if step != 0:
            assert old_img is not None
            return alpha * self.toRGBs[step](e) + (1 - alpha) * old_img
        else:
            return self.toRGBs[step](e)
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        # create from_rgb layer
        mult = 1
        ngf = 64
        from_rgb_ngf1 = [nn.ReflectionPad2d(3),
                         nn.Conv2d(input_nc, int(ngf * mult), kernel_size=7, padding=0, bias=True),
                         nn.InstanceNorm2d(int(ngf * mult), affine=True),
                         nn.ReLU(True)]
        mult = 2
        from_rgb_ngf2 = [nn.ReflectionPad2d(3),
                         nn.Conv2d(input_nc, ngf * mult, kernel_size=7, padding=0, bias=True),
                         nn.InstanceNorm2d(int(ngf * mult), affine=True),
                         nn.ReLU(True)]

        mult = 4
        from_rgb_ngf4 = [nn.ReflectionPad2d(3),
                         nn.Conv2d(input_nc, ngf * mult, kernel_size=7, padding=0, bias=True),
                         nn.InstanceNorm2d(int(ngf * mult), affine=True),
                         nn.ReLU(True)]

        l_from_rgb_ngf4 = nn.Sequential(*from_rgb_ngf4)
        l_from_rgb_ngf2 = nn.Sequential(*from_rgb_ngf2)
        l_from_rgb_ngf1 = nn.Sequential(*from_rgb_ngf1)
        self.fromRGBs = nn.ModuleList([l_from_rgb_ngf4, l_from_rgb_ngf2, l_from_rgb_ngf1])

        # down sample
        # 128
        down_sample_1 = [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]
        # 64
        down_sample_2 = [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]
        l_downsample_1 = nn.Sequential(*down_sample_1)
        l_downsample_2 = nn.Sequential(*down_sample_2)
        self.downSamples = nn.ModuleList([l_downsample_1, l_downsample_2])

        # 32
        model = [nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True)]
        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, img, step=0, alpha=0):
        e = self.fromRGBs[step](img)
        old_e = e
        if step != 0:
            old_img = self.downsample(img)
            old_e = self.fromRGBs[step - 1](old_img)
            old_step = step - 1
            for i in range(old_step - 1, -1, -1):
                old_e = self.downSamples[-1 + i](old_e)

        for i in range(step - 1, -1, -1):
            e = self.downSamples[-1 + i](e)

        if step == 0:
            e = self.model(e)
            # Average pooling and flatten
            return F.avg_pool2d(e, e.size()[2:]).view(e.size()[0], -1)
        else:
            e = alpha * self.model(e) + (1 - alpha) * self.model(old_e)
            # Average pooling and flatten
            return F.avg_pool2d(e, e.size()[2:]).view(e.size()[0], -1)