"""Models for the RFNN experiments"""
import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf


class Layers(object):
    def __init__(self, group):
        if group == "V":
            from groups import V_group
            self.group = V_group()
        elif group == "S4":
            from groups import S4_group
            self.group = S4_group()
        elif group == "T4":
            from groups import T4_group
            self.group = T4_group()
        elif group == "Z4":
            from groups import Z4_group
            self.group = Z4_group()
        elif group == "D3":
            from groups import D3_group
            self.group = D3_group()
        else:
            print("Group is not recognized")
            sys.exit(-1)
        self.group_dim = self.group.group_dim
            
        # Constants
        self.cayley = self.group.cayleytable
        

    def get_kernel(self, name, shape, factor=2.0, trainable=True):
        init = tf.contrib.layers.variance_scaling_initializer(factor=factor)
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name, shape=shape, initializer=init, trainable=trainable)
        return kernel


    def conv(self, x, kernel_size, n_out, strides=1, padding="SAME"):
        """A basic 3D convolution"""
        with tf.variable_scope("conv"):
            n_in = x.get_shape().as_list()[-1]
            W = self.get_kernel('W', [kernel_size,kernel_size,kernel_size,n_in,n_out])
            return tf.nn.conv3d(x, W, (1,strides,strides,strides,1), padding)


    def conv_block(self, x, kernel_size, n_out, is_training, use_bn=True, strides=1,
                   padding="SAME", fnc=tf.nn.relu, name="conv_block"):
        """Convolution with batch normalization/bias and nonlinearity"""
        with tf.variable_scope(name):
            y = self.conv(x, kernel_size, n_out, strides=strides, padding=padding)
            beta_init = tf.constant_initializer(0.01)
            if use_bn:
                return fnc(tf.layers.batch_normalization(y, training=is_training,
                           beta_initializer=beta_init))
            else:
                bias = tf.get_variable("bias", [n_out], initializer=beta_init)
                return fnc(tf.nn.bias_add(y, bias))


    def Gconv_block(self, x, kernel_size, n_out, is_training, use_bn=True, strides=1,
            padding="SAME", fnc=tf.nn.relu, name="Gconv_block", drop_sigma=0.1):
        """Convolution with batch normalization/bias and nonlinearity"""
        with tf.variable_scope(name):
            y = self.Gconv(x, kernel_size, n_out, is_training, strides=strides, padding=padding, drop_sigma=drop_sigma)
            beta_init = tf.constant_initializer(0.01)
            y = tf.transpose(y, perm=[0,1,2,3,5,4])
            ysh = y.get_shape().as_list()
            if use_bn:
                y = tf.layers.batch_normalization(y, training=is_training, beta_initializer=beta_init)
            else:
                bias = tf.get_variable("bias", [n_out], initializer=beta_init)
                y = tf.nn.bias_add(y, bias)
            return tf.transpose(fnc(y), perm=[0,1,2,3,5,4])


    def prepare_filters(self, W, kernel_size, n_in, in_group):
        WN = self.group.get_Grotations(W)
        # copy and rotate the filter by all group elements.
        # WN is a list of all rotated copies.
        if in_group == 1:
            WN = tf.stack(WN, -1) # stash all rotations (as a list) into the last dimension (as a tensor)
            WN = tf.reshape(WN, [kernel_size, kernel_size, kernel_size, n_in, -1]) 
            # separate n_in and n_out*out_group, as a preparation for conv3d
        elif in_group == self.group_dim:
            kernel_shape = [kernel_size, kernel_size, kernel_size, n_in, self.group_dim, -1]
            # kernel shape is explicitly needed, to divide the channel dimension of each copy into n_in, in_group, n_out
            # so that we can permute the in_group dimension of rotated copies.
            WN = self.group.get_Gpermutations(WN, kernel_shape)
            WN = tf.stack(WN, -1) # stash all copies (as a list) into the last dimension (as a tensor)
            WN = tf.reshape(WN, [kernel_size, kernel_size, kernel_size, n_in*self.group_dim, -1])
            # separate n_in*in_group and n_out*out_group, as a preparation for conv3d
        return WN


    def Gconv(self, x, kernel_size, n_out, is_training, strides=1, padding="SAME", drop_sigma=0.1):
        """Perform a discretized convolution on SO(3)

        Args:
            x: [batch_size, N0, N1, N2, n_in, group_dim/1]
            kernel_size: int for the spatial size of the kernel
            n_out: int for number of output channels
            strides: int for spatial stride length
            padding: "valid" or "same" padding
        Returns:
            [batch_size, new_height, new_width, new_depth, n_out, group_dim] tensor in G
        """
        batch_size = tf.shape(x)[0]
        with tf.variable_scope('Gconv'):
            xsh = x.get_shape().as_list()
            xN = tf.reshape(x, [batch_size, xsh[1], xsh[2], xsh[3], xsh[4]*xsh[5]]) # stash n_in and in_group_dim
            W = self.get_kernel("W", [kernel_size, kernel_size, kernel_size, xsh[4]*xsh[5]*n_out]) # stash n_in*in_group*n_out
            WN = self.prepare_filters(W, kernel_size, xsh[4], xsh[5])

            # Convolve
            # Gaussian dropout on the weights
            #WN *= (1 + drop_sigma*tf.to_float(is_training)*tf.random_normal(WN.get_shape()))

            if not (isinstance(strides, tuple) or isinstance(strides, list)):
                strides = (1,strides,strides,strides,1)
            if padding == 'REFLECT':
                padding = 'VALID'
                pad = WN.get_shape().as_list()[2] // 2
                xN = tf.pad(xN, [[0,0],[pad,pad],[pad,pad],[pad,pad],[0,0]], mode='REFLECT') 

            yN = tf.nn.conv3d(xN, WN, strides, padding)
            ysh = yN.get_shape().as_list()
            y = tf.reshape(yN, [batch_size, ysh[1], ysh[2], ysh[3], n_out, self.group_dim])
        return y


    def GconvTransposed(self, x, kernel_size, n_out, is_training, strides=1, padding="SAME", drop_sigma=0.1):
        """Perform a transposed group convolution

        Args:
            x: [batch_size, N0, N1, N2, n_in, group_dim/1]
            kernel_size: int for the spatial size of the kernel
            n_out: int for number of output channels
            strides: int for spatial stride length
            padding: "valid" or "same" padding
        Returns:
            [batch_size, new_height, new_width, new_depth, n_out, group_dim] tensor in G
        """
        batch_size = tf.shape(x)[0]
        with tf.variable_scope('GconvTransposed'):
            xsh = x.get_shape().as_list()
            xN = tf.reshape(x, [batch_size, xsh[1], xsh[2], xsh[3], xsh[4]*xsh[5]]) # stash n_in and in_group_dim
            W = self.get_kernel("W", [kernel_size, kernel_size, kernel_size, xsh[4]*xsh[5]*n_out]) # stash n_in*in_group*n_out
            WN = self.prepare_filters(W, kernel_size, xsh[4], xsh[5])
            WN = tf.transpose(WN, [0, 1, 2, 4, 3]) # switch input and output positions

            # Convolve
            # Gaussian dropout on the weights
            #WN *= (1 + drop_sigma*tf.to_float(is_training)*tf.random_normal(WN.get_shape()))
            if not (isinstance(strides, tuple) or isinstance(strides, list)):
                strides = (1,strides,strides,strides,1)
            if padding == 'REFLECT':
                padding = 'VALID'
                pad = WN.get_shape().as_list()[2] // 2
                xN = tf.pad(xN, [[0,0],[pad,pad],[pad,pad],[pad,pad],[0,0]], mode='REFLECT') 

            # Compute output shape TODO Improve this section
            output_shape = [xN.get_shape().as_list()[0]]  # batch size
            for i in range(1, 4):  # depth, height, width
                if padding == "SAME":
                    output_dim = xN.get_shape().as_list()[i] * strides[i]
                elif padding == "VALID":
                    output_dim = (xN.get_shape().as_list()[i] - 1) * strides[i] + WN.get_shape().as_list()[i - 1]
                output_shape.append(int(output_dim))
            output_shape.append(int(n_out*self.group_dim))  # output channels (actual output channels * dimensions)
            
            yN = tf.nn.conv3d_transpose(xN, WN, output_shape, strides, padding)
            ysh = yN.get_shape().as_list()
            y = tf.reshape(yN, [batch_size, ysh[1], ysh[2], ysh[3], n_out, self.group_dim])
        return y


    def Gres_block(self, x, kernel_size, n_out, is_training, use_bn=True,
                   strides=1, padding="SAME", fnc=tf.nn.relu, drop_sigma=0.1,  name="Gres_block"):
        """Residual block style 3D group convolution
        
        Args:
            x: [batch_size, height, width, n_in, group_dim/1]
            kernel_size: int for the spatial size of the kernel
            n_out: int for number of output channels
            strides: int for spatial stride length
            padding: "valid" or "same" padding
        Returns:
            [batch_size, new_height, new_width, new_depth, n_out, group_dim] tensor in G
        """
        with tf.variable_scope(name):
            with tf.variable_scope("residual_connection"):
                # Begin residual connection
                y = self.Gconv_block(x, kernel_size, n_out, is_training, use_bn=use_bn, strides=strides, 
                                     padding=padding, fnc=fnc, drop_sigma=drop_sigma, name="Gconv_blocka")
                y = self.Gconv_block(y, kernel_size, n_out, is_training, use_bn=use_bn, drop_sigma=drop_sigma,
                                     fnc=tf.identity, name="Gconv_blockb")

            with tf.name_scope("shortcut_connection"):
                # Recombine with shortcut
                # a) resize and pad input if necessary
                xsh = tf.shape(x)
                ysh = tf.shape(y)
                xksize = (1,kernel_size,kernel_size,kernel_size,1)
                xstrides = (1,strides,strides,strides,1)
                x = tf.reshape(x, tf.concat([xsh[:4],[-1,]], 0))
                x = tf.nn.avg_pool3d(x, xksize, xstrides, "SAME")
                x = tf.reshape(x, tf.concat([ysh[:4],[-1,self.group_dim]], 0))
                
                diff = n_out - x.get_shape().as_list()[-2]
                paddings = tf.constant([[0,0],[0,0],[0,0],[0,0],[0,diff],[0,0]])
                x = tf.pad(x, paddings)
            
            with tf.name_scope("combinator"):
                # b) recombine
                #return fnc(x+y)
                return x+y
