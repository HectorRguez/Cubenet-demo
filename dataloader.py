"""Data-loader using new TF Dataset class for Plankton"""
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf


class DataLoader(object):
    """Wrapper class around TF dataset pipeline"""
    
    def __init__(self, address_file, mode, batch_size, jitter, 
                 shuffle=True, buffer_size=1000, num_threads=8, N=16):
        """Create a new dataloader for the Kaggle plankton dataset

        Args:
            address_file: path to file of addresses and labels
            mode: 'train' or 'test'
            batch_size: int for number of images per batch
            n_classes: int for number of classes in dataset
            height: output image height
            width: output image width
            shuffle: bool for whether to shuffle dataset order
            buffer_size: int for number of images to store in buffer
            num_threads: int for number of CPU threads for preprocessing
        Raises:
            ValueError: If an invalid mode is passed
        """
        self.address_file = address_file
        self.jitter = jitter
        self.N = N

        # Read in data from address file and normalize it
        diel, gf = self._parse_raw(address_file, N)
        norm_diel = self._normalize_diel_max(diel, N)

        # Tensorize data
        self.diel = tf.convert_to_tensor(norm_diel, dtype=tf.float32)
        self.gf = tf.convert_to_tensor(gf, dtype=tf.float32)

        # Create TF dataset object
        data = tf.data.Dataset.from_tensor_slices((self.diel, self.gf))
        
        if mode == 'train':
            data = data.map(self._preprocess_train, num_parallel_calls=8).prefetch(100*batch_size)
        elif mode == 'test':
            data = data.map(self._preprocess_test, num_parallel_calls=8).prefetch(10*batch_size)
        else:
            raise ValueError("Invalid mode '{:s}'.".format(mode))

        # Shuffle within buffer for training
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # Minibatch
        self.data = data.batch(batch_size)

    def _parse_raw(self, address_file, _dtype = np.float64, N = 16):
        """ Function to read floats from a binary file 
                n           : Number of Green's function blocks N so that the cube is NxNxN
                block_width : Number of Green's function blocks that fit in a dielectric cube 
                len         : Data array length

                data                            | gf
                blockn*blockn*blockn            | n*n*6
                ...                             | ...
        """
        data = np.fromfile(address_file, dtype=_dtype)
        n = int(data[0])
        if(n != N):
            raise ValueError("Invalid number of blocks, N = {:i} and ggft solver N = {:i}.".format(self.N, n))
        block_width = int(data[1])
        blockn = n // block_width
        data = data[2:]
        len = blockn*blockn*blockn + n*n*6
        data = data.reshape(-1, len)
        # Split data into dielectric constants (data) and Green's function (gf)
        data, gf = np.split(data, [blockn*blockn*blockn], axis=1)

        return data, gf

    def _index(self, i, j, k, N=16):
        return int(i + j*N + k*N*N)

    def _center_index(self, N=16):
        """ Function to return the index of the center of the cube """
        nhalf = int(N // 2)
        if N % 2:
            return [
                self._index(nhalf,   nhalf,   nhalf,   N), self._index(nhalf,   nhalf,   nhalf-1, N), 
                self._index(nhalf,   nhalf-1, nhalf,   N), self._index(nhalf-1, nhalf,   nhalf,   N), 
                self._index(nhalf-1, nhalf-1, nhalf,   N), self._index(nhalf-1, nhalf,   nhalf-1, N), 
                self._index(nhalf,   nhalf-1, nhalf-1, N), self._index(nhalf-1, nhalf-1, nhalf-1, N)]
        return [self._index(nhalf, nhalf, nhalf, N)]

    def _boundary_index(self, le, N=16):
        """ Function to return the indexes of the boundary blocks of the cube """
        res = []
        def condition(i, level): 
            return (i <= level or i >= N - 1 - level)
        for k in range(N):
            for j in range(N):
                for i in range(N):
                    if condition(i, le) or condition(j, le) or condition(k, le):
                        res.append(self._index(i, j, k, N))
        return res

    def _trunk_index(self, le, N=16):
        """ Function to return the indexes of the trunk blocks of the cube """
        res = []
        def condition(i, level):
            return (i > level and i < N - 1 - level)
        for k in range(N):
            for j in range(N):
                for i in range(N):
                    if (condition(i, le) or condition(j, le)) and (condition(i, le) or condition(k, le)) and (condition(k, le) or condition(j, le)):
                        res.append(self._index(i, j, k, N))
        return res
    
    def _center_diel(self, data, N=16):
        """ Function to return the average dielectric of the cube center cube """
        return np.average(data[:, self._center_index(N)], axis=1).reshape(-1, 1)

    def _normalize_diel_max(self, data, _dtype=np.float64, N=16):
        """ Function to normalize the dielectric coefficients """
        norm_data = data / data.max(axis=1, keepdims=True)
        return norm_data


    def _preprocess_train(self, diel, gf):
        """"Input preprocessing for training mode"""

        return diel, gf


    def _preprocess_test(self, diel, gf):
        """"Input preprocessing for training mode"""
        
        return diel, gf
