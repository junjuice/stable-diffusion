from wavemix import Level4Waveblock, Level3Waveblock, Level2Waveblock, Level1Waveblock
import torch.nn as nn



class WaveMix(nn.Module):
    def __init__(
            self,
            *,
            depth = 4,
            mult = 2,
            ff_channel = 144,
            final_dim = 144,
            dropout = 0.,
            level = 1,
        ):

            super().__init__()

            self.layers = nn.ModuleList([]) 
            for _ in range(depth): 
                if level == 4:
                    self.layers.append(Level4Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))
                elif level == 3:
                    self.layers.append(Level3Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))
                elif level == 2:
                    self.layers.append(Level2Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))
                else:
                    self.layers.append(Level1Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))

                
            self.expand = nn.Sequential(
            nn.ConvTranspose2d(final_dim,int(final_dim/2), 4, stride=2, padding=1),
            nn.Conv2d(int(final_dim/2), 3, 1)
            )
            

            self.conv = nn.Sequential(
              nn.Conv2d(3, int(final_dim/2), 3, 1, 1),
              nn.Conv2d(int(final_dim/2),final_dim, 3, 1, 1)
            )
            


    def forward(self, img):
        x = self.conv(img)

        for attn in self.layers:
            x = attn(x) + x

        out = self.expand(x)
        
        return out