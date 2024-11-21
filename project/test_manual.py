import minitorch


import numpy as np
from minitorch.tensor import Tensor

# Define two matrices
matrix_a = Tensor[
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15]
]

matrix_b = Tensor[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
]

# Convert numpy arrays to minitorch Tensors
tensor_a = Tensor(matrix_a)
tensor_b = Tensor(matrix_b)

# Perform matrix multiplication using SimpleOps
result = tensor_a @ tensor_b

print(result)
