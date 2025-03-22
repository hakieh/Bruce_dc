import torch
import numpy as np

tensor_list = []

# 假设循环 10 次生成 (1, 10) 的张量
for _ in range(2):
    new_tensor = torch.randn(1, 10)  # (1, 10)
    tensor_list.append(new_tensor.squeeze(0))  # 去掉 batch 维，变为 (10,)

print(tensor_list)
# 用 stack 堆叠，结果是 (10, 10)
result_tensor = torch.stack(tensor_list, dim=0)

# 转成 numpy
final_array = result_tensor.numpy()

print(final_array.shape)  # 输出: (10, 10)
print(final_array)