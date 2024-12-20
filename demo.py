import numpy as np
import torch

from layers import GConv3D
from groups import V_group, S4_group, T4_group, D3_group, Z4_group

np.random.seed(42)
torch.manual_seed(42)

def is_close(tensor_a, tensor_b, epsilon=1e-4):
    return torch.allclose(tensor_a, tensor_b, rtol=0, atol=epsilon)

def show_all_rotations():
    kernel_size = 2
    example_filter = torch.randn(1,1,1, kernel_size, kernel_size, kernel_size)
    group = V_group()
    example_filter_rotated_copies = group.get_Grotations(example_filter)
    for i in range(4):
        print(f"V-group Rotation {i}")
        print(example_filter_rotated_copies[i])
        # test rotate_tensor_with_batch
        # manually rotate the filter by group element i
        x_rotated = group.rotate_tensor(example_filter.unsqueeze(0), element=i, start_dim=4)
        c = is_close(example_filter_rotated_copies[i], x_rotated[0])
        assert c, "test rotate_tensor_with_batch failed"
        print()
    group = T4_group()
    example_filter_rotated_copies = group.get_Grotations(example_filter)
    for i in range(12):
        print(f"T4-group Rotation {i}")
        print(example_filter_rotated_copies[i])
        # test rotate_tensor_with_batch
        # manually rotate the filter by group element i
        x_rotated = group.rotate_tensor(example_filter.unsqueeze(0), element=i, start_dim=4)
        c = is_close(example_filter_rotated_copies[i], x_rotated[0])
        assert c, "test rotate_tensor_with_batch failed"
        print()
    group = S4_group()
    example_filter_rotated_copies = group.get_Grotations(example_filter)
    for i in range(24):
        print(f"S4-group Rotation {i}")
        print(example_filter_rotated_copies[i])
        # test rotate_tensor_with_batch
        # manually rotate the filter by group element i
        x_rotated = group.rotate_tensor(example_filter.unsqueeze(0), element=i, start_dim=4)
        c = is_close(example_filter_rotated_copies[i], x_rotated[0])
        assert c, "test rotate_tensor_with_batch failed"
        print()
    group = D3_group()
    example_filter_rotated_copies = group.get_Grotations(example_filter)
    for i in range(6):
        print(f"D3-group Rotation {i}")
        print(example_filter_rotated_copies[i])
        # test rotate_tensor_with_batch
        # manually rotate the filter by group element i
        x_rotated = group.rotate_tensor(example_filter.unsqueeze(0), element=i, start_dim=4)
        c = is_close(example_filter_rotated_copies[i], x_rotated[0])
        assert c, "test rotate_tensor_with_batch failed"
        print()
    group = Z4_group()
    example_filter_rotated_copies = group.get_Grotations(example_filter)
    for i in range(4):
        print(f"Z4-group Rotation {i}")
        print(example_filter_rotated_copies[i])
        # test rotate_tensor_with_batch
        # manually rotate the filter by group element i
        x_rotated = group.rotate_tensor(example_filter.unsqueeze(0), element=i, start_dim=4)
        c = is_close(example_filter_rotated_copies[i], x_rotated[0])
        assert c, "test rotate_tensor_with_batch failed"
        print()


def show_all_rotations_permutations():
    kernel_size = 2
    example_filter = torch.randn(1,6,1, kernel_size, kernel_size, kernel_size)
    group = D3_group()
    example_filter_rotated_copies = group.get_Grotations_permutations(example_filter)
    for i in range(6):
        print(f"D3-group Rotation {i}")
        print(example_filter_rotated_copies[i])
        # test rotate_tensor_with_batch
        # manually rotate the filter by group element i
        x_rotated = group.rotate_tensor(example_filter.unsqueeze(0), element=i, start_dim=4)
        # print(f"only rotate {x_rotated}")
        x_rotated = group.permute_tensor(x_rotated, element=i, dim=2)
        c = is_close(example_filter_rotated_copies[i], x_rotated[0])
        assert c, "test rotate_tensor_with_batch failed"
        print()


def test_equivariance():
    ############################################################################################################################
    #                                                       TEST PARAMETERS
    ############################################################################################################################
    batch_size =           2
    input_cube_size =      14
    input_channel_size =   2 
    kernel_size =          4
    stride =               2            
    transposed =           False
    transposed2 =          True


    output_channel_size =  2        
            
    # group = V_group()

    # group = D3_group()
    # group = Z4_group()
    # group = T4_group()
    group = S4_group()
    input_group_size =     group.group_dim
    input_group_size =     1

    output_cube_size =     (input_cube_size - kernel_size) // stride + 1 if not transposed else (input_cube_size - 1) * stride + kernel_size
    output_cube_size2 =     (output_cube_size - kernel_size) // stride + 1 if not transposed2 else (output_cube_size - 1) * stride + kernel_size
    assert output_cube_size2 == input_cube_size

    ############################################################################################################################
    #                                                   PERFORM CONVOLUTION TEST 
    ############################################################################################################################
    x = torch.randn(batch_size, input_group_size, input_channel_size, input_cube_size, input_cube_size, input_cube_size)
    layer = torch.nn.Sequential(
        GConv3D(group, input_group_size, input_channel_size, output_channel_size, kernel_size, transposed=transposed, stride=stride, padding=0),
        GConv3D(group, group.group_dim, output_channel_size, output_channel_size, kernel_size, transposed=transposed2, stride=stride, padding=0)
    )
    # layer = GConv3D(group, input_group_size, input_channel_size, output_channel_size, kernel_size, transposed=transposed, stride=stride, padding=0)
    group_size = group.group_dim

    # Shape test
    result = layer(x)
    expected_shape = (batch_size, group_size, output_channel_size, output_cube_size2, output_cube_size2, output_cube_size2)
    assert result.shape == expected_shape, f"Gconv output shape {result.shape} does not match expected {expected_shape}"
    print("test_dim_Gconv passed")
        
    for test_element in range(group.group_dim):
        # Rotate and permute input to obtain a rotated output
        x_rotated = group.rotate_tensor(x, test_element, start_dim=3)
        if input_group_size == group.group_dim:
            x_rotated = group.permute_tensor(x_rotated, test_element, dim=1)
        assert x_rotated.shape == x.shape, f"rotate_tensor_with_batch shape mismatch"
        result_rotated = layer(x_rotated)
        print(f"Equivariance Test: test_element {test_element}")
        # print("x_rotated")
        # print(sess.run(x_rotated))
        # print("result_rotated")
        # print(sess.run(result_rotated))
        # Rotate the output to perform the check

        # result_manual = layer.group.permute_tensor(result_rotated, layer.group.inverse_map[test_element])
        # result_manual = layer.group.rotate_tensor_with_batch(result_manual, layer.group.inverse_map[test_element])
        result_manual = group.rotate_tensor(result, test_element, start_dim=3)
        # print(f"result_manual {result_manual}")
        result_manual = group.permute_tensor(result_manual, test_element, dim=1)
        # print(sess.run(result_manual))
        # print(f"Equivariance Test: rotated and permuted output does not match x_rotated result {result_rotated.shape}. \nTest element {test_element}\n {result_manual}\n vs {result_rotated}")
        assert is_close(result_manual, result_rotated), f"Equivariance Test: rotated and permuted output does not match x_rotated result {result_rotated.shape}. \nTest element {test_element}\n {result_manual}\n vs {result_rotated}" 
        


# show_all_rotations()
# show_all_rotations_permutations()
test_equivariance()
print("All tests passed")
