"""Models for the RFNN experiments"""
import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf

from layers import Layers


class GVGG(Layers):
    def __init__(self, is_training, args):
        super().__init__(args.group)

        # Constants
        self.batch_size = 32 # TODO Pass batch size as an argument

        # Input configuration
        self.nc = 1
        self.drop_sigma = 0.01


    def get_pred(self, x, is_training): #, reuse=False):
        with tf.variable_scope('prediction') as scope: #, reuse=reuse) as scope:
            init = tf.contrib.layers.variance_scaling_initializer()
            use_bn = True
            ds = self.drop_sigma

            # Encoder
            x = self.Gconv_block(x, 3, 2,  is_training, use_bn=use_bn, strides=2, drop_sigma=ds, padding='VALID', name="Gconv_1")  # Size 8
            x = self.Gconv_block(x, 3, 4,  is_training, use_bn=use_bn, strides=2, drop_sigma=ds, padding='VALID', name="Gconv_2")  # Size 4
            x = self.Gconv_block(x, 3, 8,  is_training, use_bn=use_bn, strides=2, drop_sigma=ds, padding='VALID', name="Gconv_3")  # Size 2
            x = self.Gconv_block(x, 2, 16, is_training, use_bn=use_bn, strides=1, drop_sigma=ds, padding='VALID', name="Gconv_4")  # Size 1

            # Decoder
            x = self.GconvTransposed_block(x, 2, 16, is_training, use_bn=use_bn, strides=1, drop_sigma=ds, padding='VALID', name="GconvTransposed_1") # Size 2
            x = self.GconvTransposed_block(x, 3, 8,  is_training, use_bn=use_bn, strides=2, drop_sigma=ds, padding='VALID', name="GconvTransposed_2") # Size 4
            x = self.GconvTransposed_block(x, 3, 4,  is_training, use_bn=use_bn, strides=2, drop_sigma=ds, padding='VALID', name="GconvTransposed_3") # Size 8
            x = self.GconvTransposed_block(x, 3, 1,  is_training, use_bn=use_bn, strides=2, drop_sigma=ds, padding='VALID', name="GconvTransposed_4") # Size 16
            
        return x