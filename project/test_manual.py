import minitorch

import random

import numba
import time

import minitorch

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
if numba.cuda.is_available():
    GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)

import numpy as np
# from minitorch.tensor import Tensor

# Define two matrices
matrix_a = minitorch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15]
])

matrix_b = minitorch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

# Convert numpy arrays to minitorch Tensors
# tensor_a = Tensor(matrix_a)
# tensor_b = Tensor(matrix_b)

# Perform matrix multiplication using CudaOps
result = matrix_a @ matrix_b
print(result.shape)
print(result)

# result = matrix_a * matrix_b

num_pts, out_size = matrix_a.shape[0], matrix_b.shape[1]

matrix_b = matrix_b.view(1, 3, 4)
matrix_a = matrix_a.view(5, 3, 1)

elementwise_product = matrix_a * matrix_b

# Sum along the appropriate axis (axis=1 for rows)
summed_result = elementwise_product.sum(1)

# Reshape the result to the desired shape

result = summed_result.view(num_pts, out_size)
print(result.shape)


print(result)

