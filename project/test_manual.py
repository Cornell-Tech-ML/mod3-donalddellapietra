import minitorch

import random

import numba
import time

import minitorch

import numpy as np

import matplotlib.pyplot as plt




# a = minitorch.tensor(np.array([1, 2, 3]).tolist())
# print(a.shape)
# print(a)

# FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
# if numba.cuda.is_available():
#     GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)

# backend = FastTensorBackend if not numba.cuda.is_available() else GPUBackend

# import numpy as np

# # Define two matrices
# matrix_a = minitorch.tensor([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9],
#     [10, 11, 12],
#     [13, 14, 15]
# ])

# matrix_b = minitorch.tensor([
#     [1, 2, 3, 4],
#     [5, 6, 7, 8],
#     [9, 10, 11, 12]
# ])



# num_pts, out_size = matrix_a.shape[0], matrix_b.shape[1]

# matrix_b_1 = matrix_b.view(1, 3, 4)
# matrix_a_1 = matrix_a.view(5, 3, 1)

# elementwise_product = matrix_a_1 * matrix_b_1

# summed_result = elementwise_product.sum(1)



# result = summed_result.view(num_pts, out_size)
# print(result.shape)

# matrix_a.backend = backend
# matrix_b.backend = backend
# result = matrix_a @ matrix_b
# print(result.shape)
# print(result)

# print(result)



FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
if numba.cuda.is_available():
    GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)

backend = FastTensorBackend if not numba.cuda.is_available() else GPUBackend

# Function to perform matrix multiplication using elementwise product and summing
def elementwise_multiplication(matrix_a, matrix_b):
    matrix_b_1 = matrix_b.view(1, matrix_b.shape[0], matrix_b.shape[1])
    matrix_a_1 = matrix_a.view(matrix_a.shape[0], matrix_a.shape[1], 1)
    elementwise_product = matrix_a_1 * matrix_b_1
    summed_result = elementwise_product.sum(1)
    return summed_result

# Function to perform matrix multiplication using the @ operator
def real_matrix_multiplication(matrix_a, matrix_b):
    return matrix_a @ matrix_b


def reduce_test(matrix_a):
    return matrix_a.sum(0)



# Test over 40 iterations
num_iterations = 40
elementwise_time = 0
real_time = 0
reduce_time = 0
reduce_time_cuda = 0
elementwise_times = []
real_times = []
reduce_times = []
reduce_times_cuda = []
for _ in range(num_iterations):
    # Generate random sizes between 4x4 and 100x100
    rows_a = random.randint(4, 100)
    cols_a = random.randint(4, 100)
    rows_b = cols_a
    cols_b = random.randint(4, 100)

    # Generate random matrices
    matrix_a = minitorch.tensor(np.random.rand(rows_a, cols_a).tolist())
    matrix_b = minitorch.tensor(np.random.rand(rows_b, cols_b).tolist())

    print(matrix_a.shape, matrix_b.shape)


    # Time elementwise multiplication
    start_time = time.time()
    elementwise_result = elementwise_multiplication(matrix_a, matrix_b)
    elementwise_time += time.time() - start_time
    elementwise_times.append(time.time() - start_time)

    reduce_start_time = time.time()
    reduce_result = reduce_test(matrix_a)
    reduce_time += time.time() - reduce_start_time
    reduce_times.append(time.time() - reduce_start_time)


    # Set backend
    matrix_a.backend = backend
    matrix_b.backend = backend

    # Time real matrix multiplication
    start_time = time.time()
    real_result = real_matrix_multiplication(matrix_a, matrix_b)
    real_time += time.time() - start_time
    real_times.append(time.time() - start_time)


    reduce_start_time = time.time()
    reduce_result_cuda = reduce_test(matrix_a)
    reduce_time_cuda += time.time() - reduce_start_time
    reduce_times_cuda.append(time.time() - reduce_start_time)


print(f"Average time for elementwise multiplication: {elementwise_time / num_iterations:.6f} seconds")
print(f"Average time for real matrix multiplication: {real_time / num_iterations:.6f} seconds")
print(f"Average time for reduce: {reduce_time / num_iterations:.6f} seconds")
print(f"Average time for reduce cuda: {reduce_time_cuda / num_iterations:.6f} seconds")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(range(num_iterations), elementwise_times, label='Elementwise Multiplication', color='blue')
plt.plot(range(num_iterations), real_times, label='Real Matrix Multiplication', color='red')
plt.xlabel('Iteration')
plt.ylabel('Time (seconds)')
plt.title('Comparison of Matrix Multiplication Methods')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(num_iterations), reduce_times, label='Reduce', color='blue')
plt.plot(range(num_iterations), reduce_times_cuda, label='Reduce CUDA', color='red')
plt.xlabel('Iteration')
plt.ylabel('Time (seconds)')
plt.title('Comparison of Reduce Methods')
plt.legend()
plt.show()