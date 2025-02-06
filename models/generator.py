import torch
import torch.nn as nn
import functools

# U-Net based generator for CycleGAN #
class BasicBlock(nn.Module):
    '''
    Unet conv block for generator
    '''
    def __init__(self, in_channels, out_channels, norm='batch', down=True):
        super().__init__()
        if norm =='batch':
            norm_layer = functools.partial(nn.BatchNorm2d ,affine=True, track_running_stats=True)
        elif norm == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)

        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                norm_layer(out_channels),
                nn.LeakyReLU(0.2)
        )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                norm_layer(out_channels),
                nn.ReLU()
            )
    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    '''
    Residual block for generator
    '''
    def __init__(self, channels):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.residual(x)
    
class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )

        self.down_blocks = nn.ModuleList([
            BasicBlock(features, features * 2, norm='instance', down=True),
            BasicBlock(features * 2, features * 4, norm='instance', down=True)
        ])

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(features * 4) for _ in range(num_residuals)]
        )

        self.up_blocks = nn.ModuleList([
            BasicBlock(features * 4, features * 2, norm='instance', down=False),
            BasicBlock(features * 2, features, norm='instance', down=False)
        ])

        self.last = nn.Conv2d(features, in_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect')
    
    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        x = torch.tanh(self.last(x))

        return x

x = torch.randn((1, 3, 256, 256))
gen = Generator(in_channels=3, features=64)
print(gen(x).shape)
