import numpy as np
import tensorflow as tf

from layers import Layers
from groups import V_group, S4_group, T4_group

np.random.seed(42)
tf.set_random_seed(42)


def is_close(tensor_a, tensor_b, epsilon=1e-4):
    return tf.reduce_all(tf.abs(tensor_a - tensor_b) < epsilon)

def show_all_rotations():
    kernel_size = 2
    with tf.Session() as sess:
        example_filter = np.random.rand(kernel_size, kernel_size, kernel_size, 1) # just 1 channel
        example_filter = tf.constant(example_filter, dtype=tf.float32)
        group = V_group()
        example_filter_rotated_copies = group.get_Grotations(example_filter)
        for i in range(4):
            print(f"V-group Rotation {i}")
            print(sess.run(example_filter_rotated_copies[i]))
            # test rotate_tensor_with_batch
            # manually rotate the filter by group element i
            x_rotated = group.rotate_tensor_with_batch(tf.expand_dims(example_filter, axis=0), i)
            c = is_close(example_filter_rotated_copies[i], x_rotated)
            assert sess.run(c), "test rotate_tensor_with_batch failed"
            # print(f"test rotate_tensor_with_batch")
            # print(sess.run(c))
            print()

        group = T4_group()
        example_filter_rotated_copies = group.get_Grotations(example_filter)
        for i in range(12):
            print(f"T4-group Rotation {i}")
            print(sess.run(example_filter_rotated_copies[i]))
            # test rotate_tensor_with_batch
            # manually rotate the filter by group element i
            x_rotated = group.rotate_tensor_with_batch(tf.expand_dims(example_filter, axis=0), i)
            c = is_close(example_filter_rotated_copies[i], x_rotated)
            assert sess.run(c), "test rotate_tensor_with_batch failed"
            # print(f"test rotate_tensor_with_batch")
            # print(sess.run(c))
            print()

        group = S4_group()
        example_filter_rotated_copies = group.get_Grotations(example_filter)
        for i in range(24):
            print(f"S4-group Rotation {i}")
            print(sess.run(example_filter_rotated_copies[i]))
            # test rotate_tensor_with_batch
            # manually rotate the filter by group element i
            x_rotated = group.rotate_tensor_with_batch(tf.expand_dims(example_filter, axis=0), i)
            c = is_close(example_filter_rotated_copies[i], x_rotated)
            assert sess.run(c), "test rotate_tensor_with_batch failed"
            # print(f"test rotate_tensor_with_batch")
            # print(sess.run(c))
            print()


def test_equivariance():
    ############################################################################################################################
    #                                                       TEST PARAMETERS
    ############################################################################################################################
    batch_size =           2
    input_cube_size =      5
    input_channel_size =   2 
    kernel_size =          3
    output_cube_size =     3
    stride =               1            

    input_group_size =     24
    output_channel_size =  2        
            
    group = "S4"

    ############################################################################################################################
    #                                                   PERFORM CONVOLUTION TEST 
    ############################################################################################################################
    x = np.random.rand(batch_size, input_cube_size, input_cube_size, input_cube_size, input_channel_size, input_group_size)
    x = tf.constant(x, dtype=tf.float32)

    layer = Layers(group)
    group_size = layer.group.group_dim
    # Shape test
    result = layer.Gconv(x, kernel_size=kernel_size, n_out=output_channel_size, strides=stride, is_training=False, padding='VALID')
    expected_shape = [batch_size, output_cube_size, output_cube_size, output_cube_size, output_channel_size, group_size]
    assert result.shape == expected_shape, f"Gconv output shape {result.shape} does not match expected {expected_shape}"
    print("test_dim_Gconv passed")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # print("original x")
        # print(sess.run(x))
        # print("original result")
        # print(sess.run(result))
        
        for test_element in range(layer.group.group_dim):
            # Rotate and permute input to obtain a rotated output
            x_rotated = layer.group.rotate_tensor_with_batch(x, test_element)
            if input_group_size == layer.group.group_dim:
                x_rotated = layer.group.permute_tensor(x_rotated, test_element)
            assert x_rotated.shape == x.shape, f"rotate_tensor_with_batch shape mismatch"
            result_rotated = layer.Gconv(x_rotated, kernel_size=kernel_size, n_out=output_channel_size, strides=stride, is_training=False, padding='VALID')
            print(f"Equivariance Test: test_element {test_element}")
            # print("x_rotated")
            # print(sess.run(x_rotated))
            # print("result_rotated")
            # print(sess.run(result_rotated))
            # Rotate the output to perform the check

            # result_manual = layer.group.permute_tensor(result_rotated, layer.group.inverse_map[test_element])
            # result_manual = layer.group.rotate_tensor_with_batch(result_manual, layer.group.inverse_map[test_element])

            result_manual = layer.group.rotate_tensor_with_batch(result, test_element)
            result_manual = layer.group.permute_tensor(result_manual, test_element)
            # print("result_manual")
            # print(sess.run(result_manual))
            assert sess.run(is_close(result_manual, result_rotated)), f"Equivariance Test: rotated and permuted output does not match x_rotated result. Test element {test_element}\n {sess.run(result_manual)}\n vs {sess.run(result_rotated)}" 
            

# show_all_rotations()
test_equivariance()
print("All tests passed")