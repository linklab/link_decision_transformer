import torch

# 예시 텐서 생성
tensor = torch.ones(5, 10, 3)  # shape: (5, 10, 32)
for i in range(5):
    for j in range(10):
        tensor[i][j] += j

print(tensor)
# 선택할 인덱스
indices = [0, 3, 9, 2, 0]

# 텐서를 슬라이싱
transposed_tensor = tensor.transpose(1, 0) # (10, 5, 3)
sliced_tensor = transposed_tensor[indices, :, :]  # shape: (5, 5, 32)
print(sliced_tensor)
sliced_tensor = sliced_tensor[:, 0, :]
print(sliced_tensor)
# 슬라이싱된 텐서를 두 번째 축을 따라 결합하여 (5, 32) 형태로 만듭니다.
# 여기서 선택된 텐서를 모두 이어붙여 최종적으로 (5, 32) 형태로 만들어야 합니다.
# 필요한 경우 차원 변경 없이 결합할 수도 있습니다.

# 필요한 경우 차원 변경 없이 결합할 수도 있습니다.
sliced_tensor_combined = sliced_tensor.view(5, -1)

print(sliced_tensor_combined.shape)  # expected shape: (5, 32)
