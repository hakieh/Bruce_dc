import torch

a = torch.randn((2,1))
b = torch.randn((2,1))
print(a)

print(b)

c = torch.as_tensor([True,False]).unsqueeze(1)
print(c.shape)
d = torch.where(c,a,b)
print(d)