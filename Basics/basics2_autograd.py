'''
    In this code, basics of PyTorch are explored such as: autograd.
    This is inspired by: https://www.kdnuggets.com/2018/11/introduction-pytorch-deep-learning.html
'''
import torch

sampleTensor = torch.tensor([23.0, 45.2], requires_grad=True)
sampleTensor2 = torch.tensor([22.3, 4.5], requires_grad=False)
tensorSum = sampleTensor + sampleTensor2
print("Tensor sum: ", tensorSum)
# To compute the derivatives, we call the .backword() on the tensor
tensorSumSingle = (tensorSum*8).sum()
ders = tensorSumSingle.backward()
print("Derivatives: ", ders)
print("Gradient: ", sampleTensor.grad)
print(sampleTensor2.grad) # None because requires_grad =  False