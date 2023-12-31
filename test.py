from wavemix import Level1Waveblock, Level2Waveblock, Level3Waveblock, Level4Waveblock
from torch import nn
import torch

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3
    
def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class WaveBlock(nn.Module):
    def __init__(self, in_ch, out_ch, level: int, mult: int=2):
        super().__init__()
        assert level <= 4
        if level == 1:
            wavemodule = Level1Waveblock
        elif level == 2:
            wavemodule = Level2Waveblock
        elif level == 3:
            wavemodule = Level3Waveblock
        elif level == 4:
            wavemodule = Level4Waveblock
        else:
            raise NotImplementedError
        self.wavemix = wavemodule(mult=mult, ff_channel=out_ch, final_dim=out_ch)
        if in_ch != out_ch:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.conv = None

    def forward(self, x):
        if self.conv:
            x = self.conv(x)
        x = self.wavemix(x) + x
        return x

        
class ResBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))

def Decoder():
    return nn.Sequential(
        Clamp(), conv(4, 64), nn.ReLU(),
        WaveBlock(64, 64, 1), ResBlock(64, 64), ResBlock(64, 64), nn.Upsample(scale_factor=2, mode="bilinear"), conv(64, 64, bias=False),
        WaveBlock(64, 64, 2), WaveBlock(64, 64, 2), ResBlock(64, 64), nn.Upsample(scale_factor=2, mode="bilinear"), conv(64, 64, bias=False),
        WaveBlock(64, 64, 3), WaveBlock(64, 64, 3), ResBlock(64, 64), nn.Upsample(scale_factor=2, mode="bilinear"), conv(64, 64, bias=False),
        WaveBlock(64, 64, 3), conv(64, 3),
    )