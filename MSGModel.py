import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch
from torchvision.utils import save_image

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, stride=1, padding=1, bias=True, transpose=False, spectral=False, relu=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d if transpose else nn.Conv2d
        self.conv = self.conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias
            )
        if spectral:
            self.conv = spectral_norm(self.conv)
        self.useRelu = relu
        self.LeakyReLu = nn.LeakyReLU(0.2)

    def forward(self, x):
        y = self.conv(x)
        if (self.useRelu):
            y = self.LeakyReLu(y)
        return y
        

class UpsampleLayer(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor)
        return x


class GeneratorFinal(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=(1, 1), bias=True),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # 512x1x1 -> 512x4x4
        self.initial = nn.Sequential(
            ConvBlock(in_channels=512, out_channels=512, kernel_size=4, padding=0, bias=False, transpose=True, spectral=True),
            ConvBlock(in_channels=512, out_channels=512, spectral=True),
        )

        # 512x4x4 -> 512x8x8
        self.g1 = nn.Sequential(
            UpsampleLayer(scale_factor=2),
            ConvBlock(in_channels=512, out_channels=512, spectral=True),
            ConvBlock(in_channels=512, out_channels=512, spectral=True),
        )

        # 512x8x8 -> 512x16x16
        self.g2 = nn.Sequential(
            UpsampleLayer(scale_factor=2),
            ConvBlock(in_channels=512, out_channels=512, spectral=True),
            ConvBlock(in_channels=512, out_channels=512, spectral=True),
        )

        # 512x16x16 -> 512x32x32
        self.g3 = nn.Sequential(
            UpsampleLayer(scale_factor=2),
            ConvBlock(in_channels=512, out_channels=512, spectral=True),
            ConvBlock(in_channels=512, out_channels=512, spectral=True),
        )

        # 512x32x32 -> 256x64x64
        self.g4 = nn.Sequential(
            UpsampleLayer(scale_factor=2),
            ConvBlock(in_channels=512, out_channels=256, spectral=True),
            ConvBlock(in_channels=256, out_channels=256, spectral=True),

        )

        # 256x64x64 -> 128x128x128
        self.g5 = nn.Sequential(
            UpsampleLayer(scale_factor=2),
            ConvBlock(in_channels=256, out_channels=128, spectral=True),
            ConvBlock(in_channels=128, out_channels=128, spectral=True),
        )

        # 128x128x128 -> 64x256x256
        self.g6 = nn.Sequential(
            UpsampleLayer(scale_factor=2),
            ConvBlock(in_channels=128, out_channels=64, spectral=True),
            ConvBlock(in_channels=64, out_channels=64, spectral=True),
        )

        self.layers = [self.initial, self.g1, self.g2, self.g3, self.g4, self.g5, self.g6]

        self.output_layers = [
            GeneratorFinal(512),
            GeneratorFinal(512),
            GeneratorFinal(512),
            GeneratorFinal(512),
            GeneratorFinal(256),
            GeneratorFinal(128),
            GeneratorFinal(64)
        ]


    def forward(self, x):
        outputs = []
        for i in range(len(self.output_layers)):
            x = self.layers[i](x)
            outputs.append(self.output_layers[i](x))
        outputs.reverse()
        return outputs


class MiniBatchStd(nn.Module):
    def __init__(self, sigma=1e-8):
        super().__init__()
        self.sigma = sigma
        
    def forward(self, x):
        batch_std = torch.sqrt(torch.var(x, dim=0, keepdim=True) + self.sigma)
        mean_std = torch.mean(batch_std, dim=1, keepdim=True)
        mean_std = mean_std.expand([x.shape[0], *mean_std.shape[1:]])

        return torch.cat([x, mean_std], dim=1)
    

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # 3x256x256 -> 128x128x128
        self.initial = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=64, kernel_size=(1, 1), padding=0),
            MiniBatchStd(),
            ConvBlock(in_channels=65, out_channels=64),
            ConvBlock(in_channels=64, out_channels=128),
            nn.AvgPool2d(2)
        )

        # 128x128x128 : 3x128x128 = 131x128x128 -> 256x64x64
        self.d1 = nn.Sequential(
            MiniBatchStd(),
            ConvBlock(in_channels=132, out_channels=128),
            ConvBlock(in_channels=128, out_channels=256),
            nn.AvgPool2d(2)
        )

        # 256x64x64 : 3x64x64 = 259x64x64 -> 512x32x32
        self.d2 = nn.Sequential(
            MiniBatchStd(),
            ConvBlock(in_channels=260, out_channels=256),
            ConvBlock(in_channels=256, out_channels=512),
            nn.AvgPool2d(2)
        )
        
        # 512x32x32 : 3x32x32 = 515x32x32 -> 512x16x16
        self.d3 = nn.Sequential(
            MiniBatchStd(),
            ConvBlock(in_channels=516, out_channels=512),
            ConvBlock(in_channels=512, out_channels=512),
            nn.AvgPool2d(2)
        )

        # 512x16x16 : 3x16x16 = 515x16x16 -> 512x8x8
        self.d4 = nn.Sequential(
            MiniBatchStd(),
            ConvBlock(in_channels=516, out_channels=512),
            ConvBlock(in_channels=512, out_channels=512),
            nn.AvgPool2d(2)
        )

        # 512x8x8 : 3x8x8 = 515x8x8 -> 512x4x4
        self.d5 = nn.Sequential(
            MiniBatchStd(),
            ConvBlock(in_channels=516, out_channels=512),
            ConvBlock(in_channels=512, out_channels=512),
            nn.AvgPool2d(2)
        )

        # 512x4x4 : 3x4x4 = 515x4x4 -> 1x1x1
        self.d6 = nn.Sequential(
            MiniBatchStd(),
            ConvBlock(in_channels=516, out_channels=512),
            ConvBlock(in_channels=512, out_channels=512, kernel_size=(4, 4),padding=0),
            nn.Flatten(),
            nn.Linear(512, 1)
        )
        self.layers = [self.initial, self.d1, self.d2, self.d3, self.d4, self.d5, self.d6]


    def forward(self, x):
        y = self.initial(x[0])
        for i in range(1, len(self.layers)):
            y = torch.cat([y, x[i]], dim=1)
            y = self.layers[i](y)

        return y


class DownSampleImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgPool = nn.AvgPool2d(2)

    def forward(self, x):
        shape = x.shape
        assert (len(shape) >= 3 and shape[-1] == shape[-2] and shape[-1] != 0 and (shape[-1] & (shape[-1]-1) == 0)), \
            "Image size not allowed"
        output = [x]
        while x.shape[-1] != 4:
            x = self.avgPool(x)
            output.append(x)
        return output


if __name__ == '__main__':
    x = torch.rand(3, 512, 1, 1)
    G = Generator()
    D = Discriminator()

    # Test generator shape
    print("Test Generator model intermediate output shapes")
    layers = G(x)
    for layer in layers:
        print("\t" + str(layer.shape))

    print("\nTest Discriminator model from Generator output")
    z = D(layers)
    print("\t" + str(z.shape))

    print("\nTest image downsampler")
    downsampler = DownSampleImage()
    realset = torch.rand(3, 3, 256, 256)
    downsampled_realset = downsampler(realset)
    for layer in downsampled_realset:
        print("\t" + str(layer.shape))

    print("\nTest Discriminator model from downsampled images")
    z = D(downsampled_realset)
    print("\t" + str(z.shape))