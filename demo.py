"""Models for the RFNN experiments"""
import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf

from layers import Layers

def print_group_equivariant(result, n):
    corners = [[0,0], [0,n-1], [n-1,0], [n-1,n-1]]
    for i in range(4):
        for j in range(4):
            print(f"BATCH = 0, H = {corners[j][0]}, W = {corners[j][1]}, D = 0, CHANNEL = 0, GROUP = {i}: {result[0,corners[j][0],corners[j][1],0,0,i]}")

def print_input_corners(input, n):
    corners = [[0,0], [0,n-1], [n-1,0], [n-1,n-1]]
    for i in range(4):
        print(f"BATCH = 0, H = {corners[i][0]}, W = {corners[i][1]}, D = 0, CHANNEL = 0, GROUP = 0: {input[0,corners[i][0],corners[i][1],0,0,0]}")


group = "S4"
layer = Layers(group)

x = tf.random.normal([1, 16, 16, 16, 3, 1])  # Example input tensor
    
# Shape test
result = layer.Gconv(x, kernel_size=3, n_out=6, strides=2, is_training=True)
expected_shape = [1, 8, 8, 8, 6, 4]
assert result.shape == expected_shape, f"Gconv output shape {result.shape} does not match expected {expected_shape}"
print("test_dim_Gconv passed")

# Rotate input
xsh = x.get_shape().as_list()
x_rotated = tf.reshape(x, [16, 16, -1])
x_rotated = tf.image.rot90(x_rotated, 0)
x_rotated = tf.reshape(x_rotated, xsh)
result_rotated = layer.Gconv(x_rotated, kernel_size=3, n_out=6, strides=2, is_training=True)

# Equivariance test
print("INPUT")
print_input_corners(x, 16)
print("\n")

print("INPUT ROTATED")
print_input_corners(x_rotated, 16)
print("\n")

print("RESULT")
print_group_equivariant(result, 8)
print("\n")

print("RESULT ROTATED")
print_group_equivariant(result_rotated, 8)
print("\n")