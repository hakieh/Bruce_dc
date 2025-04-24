
import torch

def calculate_row_dot_product(tensor1, tensor2):
    # 计算每行的内积 (点积)
    dot_product = torch.sum(tensor1 * tensor2, dim=1)  # 对每行的元素进行乘法并求和
    return dot_product.unsqueeze(1)  # 返回一个 n*1 的张量

# 示例：假设 tensor1 和 tensor2 为 n*3 的张量
tensor1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
tensor2 = torch.tensor([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])

result = calculate_row_dot_product(tensor1, tensor2)
print(result)
