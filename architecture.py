import torch
import torch.nn as nn
from layers import Layers

class GVGG(nn.Module):
    def __init__(self, is_training, args):
        super(GVGG, self).__init__()
        
        # Constants
        self.batch_size = 32  # TODO: Pass batch size as an argument
        
        # Input configuration
        self.nc = 1
        self.drop_sigma = 0.01

        # Encoder layers
        self.encoder = nn.Sequential(
            self.Gconv_block(1, 3, 2, is_training, use_bn=True, strides=2, drop_sigma=self.drop_sigma),
            self.Gconv_block(3, 4, 2, is_training, use_bn=True, strides=2, drop_sigma=self.drop_sigma),
            self.Gconv_block(4, 8, 2, is_training, use_bn=True, strides=2, drop_sigma=self.drop_sigma),
            self.Gconv_block(8, 16, 1, is_training, use_bn=True, strides=1, drop_sigma=self.drop_sigma),
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            self.GconvTransposed_block(16, 16, 1, is_training, use_bn=True, strides=1, drop_sigma=self.drop_sigma),
            self.GconvTransposed_block(16, 8, 2, is_training, use_bn=True, strides=2, drop_sigma=self.drop_sigma),
            self.GconvTransposed_block(8, 4, 2, is_training, use_bn=True, strides=2, drop_sigma=self.drop_sigma),
            self.GconvTransposed_block(4, 1, 2, is_training, use_bn=True, strides=2, drop_sigma=self.drop_sigma),
        )

    def forward(self, x, is_training):
        # Encoder forward pass
        x = self.encoder(x)
        
        # Decoder forward pass
        x = self.decoder(x)
        
        return x
