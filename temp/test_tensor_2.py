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
sliced_tensor = tensor[:, indices, :]  # shape: (5, 5, 3)

# 두 번째 축을 따라 인덱싱된 텐서를 첫 번째 축(batch size)을 유지하면서 (5, 32) 형태로 변경
sliced_tensor_combined = sliced_tensor.view(-1, 5, 3)[0, :, :]  # shape: (5, 32)

print(sliced_tensor_combined.shape)  # expected shape: (5, 32)
print(sliced_tensor_combined)  # 실제 슬라이싱된 텐서 출력