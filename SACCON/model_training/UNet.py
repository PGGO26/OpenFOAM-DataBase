# 2025/05/12
# 手動調整每層的結構

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_additional_inputs, dropout_rate):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_additional_intputs = num_additional_inputs
        self.dropout_rate = dropout_rate
        self.pool_ratio = 4
        self.bottom_pool_ratio = 2

        self.enc1 = self.encoder_blcok(in_channels, 4)
        self.enc2 = self.encoder_blcok(4, 16)
        self.enc3 = self.encoder_blcok(16, 64)
        self.enc4 = self.encoder_blcok(64, 256)

        self.botin1 = self.bottom_input(num_additional_inputs, 8)
        self.botin2 = self.bottom_input(8, 32)
        self.botin3 = self.bottom_input(32, 128)
        self.botin4 = self.bottom_input(128, 256)

        self.bot1 = self.bottom_block(512, 512)

        self.decbot = self.decoder_block(512, 256, pool_ratio=self.bottom_pool_ratio)
        self.dec4 = self.decoder_block(512, 64, pool_ratio=self.pool_ratio)
        self.dec3 = self.decoder_block(128, 16, pool_ratio=self.pool_ratio)
        self.dec2 = self.decoder_block(32, 4, pool_ratio=self.pool_ratio)
        self.dec1= self.decoder_block(8, out_channels, pool_ratio=self.pool_ratio)

    def encoder_blcok(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),

            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            
            nn.MaxPool2d(kernel_size=self.pool_ratio, stride=self.pool_ratio),
            nn.Dropout2d(self.dropout_rate)
        )
    
    def bottom_input(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(self.dropout_rate)
        )
    
    def bottom_block(self, in_channels, out_channels):
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
                # nn.BatchNorm2d(out_channels),
                # nn.ReLU(inplace=True),

                nn.Dropout2d(self.dropout_rate)
        )
    
    def decoder_block(self, in_channels, out_channels, pool_ratio):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=pool_ratio, stride=pool_ratio),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),

            nn.Dropout2d(self.dropout_rate)
        )
    
    def forward(self, x, mach, aoa):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Bottleneck for additional inputs expanding
        additional_inputs = torch.cat((mach, aoa), dim=1).float()
        # print(f"Additional input dtype : {additional_inputs.dtype}, size : {additional_inputs.size()}")
        bottom_intputs = self.botin1(additional_inputs)
        bottom_intputs = self.botin2(bottom_intputs)
        bottom_intputs = self.botin3(bottom_intputs)
        bottom_intputs = self.botin4(bottom_intputs)
        bottom_intputs = bottom_intputs.unsqueeze(2).unsqueeze(3).expand(-1, -1, enc4.shape[2] // self.bottom_pool_ratio, enc4.shape[3] // self.bottom_pool_ratio)
        # print(f"Bottom input size : {bottom_intputs.size()}")

        # Bottleneck for additional inputs combines Encoder outputs
        bottom = torch.cat((F.max_pool2d(enc4, self.bottom_pool_ratio), bottom_intputs), dim=1)
        assert bottom.dim() == 4, f"Unexpected bottom shape: {bottom.shape}"
        bottom = self.bot1(bottom)

        x = self.decbot(bottom)
        x = self.dec4(torch.cat((enc4, x), dim=1))
        x = self.dec3(torch.cat((enc3, x), dim=1))
        x = self.dec2(torch.cat((enc2, x), dim=1))
        y = self.dec1(torch.cat((enc1, x), dim=1))
        # print(f"Final output size : {y.size()}")

        return y