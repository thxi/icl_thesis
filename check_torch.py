import torch
x = torch.rand(5, 3)
print(x)
print(torch.cuda.is_available())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
x.to(device)
print(f'{device=}')
print(f'{x.device=}')
print('force cuda')
y = x.to('cuda')
print(f'{x.device=}')
print(f'{y.device=}')
input('waiting for input')

