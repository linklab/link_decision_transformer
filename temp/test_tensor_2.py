import torch

# 예시 텐서 생성
tensor = torch.ones(5, 10, 3)  # shape: (5, 10, 32)
for i in range(5):
    for j in range(10):
        tensor[i][j] += j

print(tensor, tensor.shape)

# 선택할 인덱스
indices = torch.tensor([0, 3, 9, 2, 0]).view(-1, 1, 1)
print(indices, indices.shape)
indices = indices.expand(size=(-1, -1, tensor.size(dim=2)))
print(indices, indices.shape)

# 텐서를 슬라이싱
sliced_tensor = tensor.gather(dim=1, index=indices)

print(sliced_tensor, sliced_tensor.shape)