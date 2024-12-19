# 2024/10/2 UNet architecture
# Total parameters : 381M
# 4 Layers, maxpool 4 for upsampling
# sampling channels : 1, 2, 8, 32, 128, 256
# Encoder 5 3*3 conv
# base channel : 32

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_additional_inputs, base_channels=2, res=256):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_additional_inputs = num_additional_inputs
        self.base_channels = base_channels
        self.layer = int(torch.log2(torch.tensor(res, dtype=torch.float32)).item()) // 2

        # Layers
        self.encoder_layers = nn.ModuleList(self.encoder(in_channels if i == 0 else base_channels*2**(2*(i-1)), base_channels*2**(2*i)) for i in range(self.layer))
        self.bottle_layers = nn.ModuleList(self.bottle_neck(num_additional_inputs if i == 0 else base_channels*2**(2*(i-1)), base_channels*2**(2*i)) for i in range(self.layer))
        self.bottle_conv = self.conv_block(base_channels*2**(2*self.layer-1), base_channels*2**(2*self.layer-1))
        self.decoder_layers = nn.ModuleList(self.decoder(base_channels*2**(2*i-1) if i == self.layer else base_channels*2**(2*i), base_channels*2**(2*(i-1))) for i in range(self.layer, 0, -1))
        self.decoder_conv = nn.ModuleList(self.conv_block(base_channels*2**(2*i-1), base_channels*2**(2*(i-1))) for i in range(self.layer, 0, -1))
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def encoder(self, in_channels, out_channels):
        return nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True)
                            )

    def bottle_neck(self, in_channels, out_channels):
        return nn.Sequential(
                            nn.Linear(in_channels, out_channels),
                            nn.BatchNorm1d(out_channels),
                            nn.ReLU(inplace=True)
                            )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True)
                            )
    
    def decoder(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=4)
    
    def forward(self, x, mach, aoa):
        # print(f"Total layer : {range(self.layer)}")
        enc_outs = []

        # Encoder path
        for i, enc in enumerate(self.encoder_layers):
            x = enc(x) if i == 0 else enc(F.max_pool2d(x, 4))
            enc_outs.append(x)
            # print(f'Layer : {i}, size : {x.size()}')

        # Bottleneck with additional inputs
        additional_inputs = torch.cat((mach, aoa), dim=1)
        for i, bottle in enumerate(self.bottle_layers):
            bottom = bottle(additional_inputs) if i == 0 else bottle(bottom)
            # print(f"Bottom input size : {bottom.size()}")
        bottom = bottom.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.shape[2] // 4, x.shape[3] // 4)
        x = torch.cat((F.max_pool2d(x, 4), bottom), dim=1)
        x = self.bottle_conv(x)
        # print(f"Bottom output size : {x.size()}")

        # Decoder path
        for i in range(self.layer):
            x = self.decoder_layers[i](x)
            # print(f"Layer : {-(i-self.layer+1)}, x enc input size : {x.size()}, {enc_outs[-(i-self.layer+1)].size()}")
            x = torch.cat((enc_outs[-(i-self.layer+1)], x), dim=1)
            x = self.decoder_conv[i](x)
            # print(f"Layer : {-(i-self.layer+1)}, dec output size : {x.size()}")
        
        # Output layer
        return self.final_conv(x)
