"""Models for the RFNN experiments"""
import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf

from layers import Layers


with tf.Session() as sees:
    def print_group_equivariant(result, n, channel, output_group_size):
        corners = [[0,0], [0,n-1], [n-1,0], [n-1,n-1]]
        for i in range(output_group_size):
            for j in range(4):
                print(f"BATCH = 0, H = {corners[j][0]}, W = {corners[j][1]}, D = 0, CHANNEL = {channel}, GROUP = {i}: {(result[0,corners[j][0],corners[j][1],0,channel,i])}")

    def print_input_corners(input, n, channel):
        corners = [[0,0], [0,n-1], [n-1,0], [n-1,n-1]]
        for i in range(4):
            print(f"BATCH = 0, H = {corners[i][0]}, W = {corners[i][1]}, D = 0, CHANNEL = {channel}, GROUP = 0: {(input[0,corners[i][0],corners[i][1],0,channel,0])}")

    ############################################################################################################################
    #                                                       TEST PARAMETERS
    ############################################################################################################################
    batch_size =           2
    input_cube_size =      5
    input_channel_size =   3 
    kernel_size =          3
    output_cube_size =     3
    stride =               1      

    input_group_size =     1        # TODO: Have not checked this yet
    output_channel_size =  4        
    output_group_size =    24       
            
    tolerance = 1e-4
    group = "S4"
    layer = Layers(group)
    cayley = layer.group.get_cayleytable()
    rotation = np.pi/2 
    perm_mat = layer.group.get_permutation_matrix(cayley, 3) # cayley table index
    print_tensors = False
    print_tensors_channel = 1

    ############################################################################################################################
    #                                                   PERFORM CONVOLUTION TEST 
    ############################################################################################################################
    x = np.random.rand(batch_size, input_cube_size, input_cube_size, input_cube_size, input_channel_size, input_group_size)
    x = tf.constant(x, dtype=tf.float32)

    # Shape test
    result = layer.Gconv(x, kernel_size=kernel_size, n_out=output_channel_size, strides=stride, is_training=False, padding='VALID')
    expected_shape = [batch_size, output_cube_size, output_cube_size, output_cube_size, output_channel_size, output_group_size]
    assert result.shape == expected_shape, f"Gconv output shape {result.shape} does not match expected {expected_shape}"
    print("test_dim_Gconv passed")

    # Rotate input to obtain a rotated output
    x_shape = x.get_shape().as_list()
    x_rotated = tf.reshape(x, [batch_size, input_cube_size, input_cube_size, -1])
    x_rotated = tf.contrib.image.rotate(x_rotated, rotation)
    x_rotated = tf.reshape(x_rotated, x_shape)
    result_rotated = layer.Gconv(x_rotated, kernel_size=kernel_size, n_out=output_channel_size, strides=stride, is_training=False, padding='VALID')

    # Permute the output to perform the check
    result_rotated_sh = result_rotated.get_shape().as_list()
    w = tf.reshape(result_rotated, [-1, output_group_size])
    w = w @ perm_mat
    result_rotated_permuted = tf.reshape(w, result_rotated_sh)

    # Rotate the permuted output to perform the check
    result_rotated_permuted_rotated_back = tf.reshape(result_rotated_permuted, [batch_size, output_cube_size, output_cube_size, -1])
    result_rotated_permuted_rotated_back = tf.contrib.image.rotate(result_rotated_permuted_rotated_back, -rotation)
    result_rotated_permuted_rotated_back = tf.reshape(result_rotated_permuted_rotated_back, result_rotated_sh)

    # Tensor value printing
    if(print_tensors): 
        print("INPUT")
        print_input_corners(x, input_cube_size, print_tensors_channel)
        print("\n")

        print("INPUT ROTATED")
        print_input_corners(sees.run(x_rotated), input_cube_size, print_tensors_channel)
        print("\n")

        print("RESULT")
        print_group_equivariant(sees.run(result), output_cube_size, print_tensors_channel, output_group_size)
        print("\n")

        # print("RESULT ROTATED")
        # print_group_equivariant(sees.run(result_rotated), output_cube_size, print_tensors_channel, output_group_size)
        # print("\n")
        # 
        # print("RESULT ROTATED PERMUTED")
        # print_group_equivariant(sees.run(result_rotated_permuted), output_cube_size, print_tensors_channel, output_group_size)
        # print("\n")

        print("RESULT ROTATED PERMUTED ROTATED")
        print_group_equivariant(sees.run(result_rotated_permuted_rotated_back), output_cube_size, print_tensors_channel, output_group_size)
        print("\n")

    # Compare the two outputs with a specific tolerance
    difference = tf.abs(result_rotated_permuted_rotated_back - result)
    within_tolerance = tf.reduce_all(tf.less_equal(difference, tolerance))

    # Evaluate the boolean result of all_equal
    all_equal_result = sees.run(within_tolerance)
    assert all_equal_result, f"Gconv output for rotated input does not match output for original input"
    print("test_equivariance_Gconv passed")