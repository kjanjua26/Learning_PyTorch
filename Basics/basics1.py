'''
    In this code, basics of PyTorch are explored such as tensors.
    This is inspired by: https://www.kdnuggets.com/2018/11/introduction-pytorch-deep-learning.html
'''
import torch

# This is for the CPU variant. 
# We initialize basic tensors and perform linear operations on them.
sampleTensor = torch.FloatTensor([[1, 2, 3], [4, 5, 7]])
sampleTensor2 = torch.FloatTensor([[20, 30, 40], [50, 60, 70]])
out = sampleTensor + sampleTensor2
print("Out: ", out)

# Initialization of a random matrix of shape (3,3)
mat3x3 = torch.randn(3,3)
mat3x3_2 = torch.randn_like(mat3x3) # to generate another random matrix like mat3x3
print("Matrix: ", mat3x3)
print("Matrix: ", mat3x3_2)
out = mat3x3 * mat3x3_2
print("Out: ", out)
transposed_out = out.t()
print("Transposed Out: ", transposed_out)