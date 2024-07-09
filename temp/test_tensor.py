import torch

# 초기 텐서
data = torch.tensor([[10, 20, 30],
                     [40, 50, 60],
                     [70, 80, 90]])

# 인덱스 텐서
indices = torch.tensor([[0],
                        [1],
                        [2]])

# gather 함수 적용
result = data.gather(dim=1, index=indices)

print(result)