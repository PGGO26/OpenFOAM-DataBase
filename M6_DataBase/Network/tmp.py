import torch
from Models.UNet_test import UNet

# 定義模型參數
in_channels = 1
out_channels = 1
num_additional_inputs = 2
base_channels = 2
res = 256

# 創建模型
model = UNet(in_channels, out_channels, num_additional_inputs, base_channels, res)

# 檢查是否有可用的 GPU，並將模型轉移到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 創建測試輸入張量
batch_size = 4  # 假設 batch size 是 4
height, width = 256, 256  # 圖像大小
x = torch.randn(batch_size, in_channels, height, width).to(device)
mach = torch.randn(batch_size, num_additional_inputs//2).to(device)  # `mach` 是一維張量
aoa = torch.randn(batch_size, num_additional_inputs//2).to(device)   # `aoa` 是一維張量

# 將數據傳入模型進行前向傳播
output = model(x, mach, aoa)

# 打印輸出形狀，確認是否正常運行
print("Output shape:", output.shape)

# import torch
# import torch.nn as nn
# from Models.UNet_test import UNet

# in_channels = 1
# out_channels = 1
# num_additional_inputs = 2  # 假設有 mach 和 aoa 兩個附加變量
# base_channels = 2  # 可以根據你的需要設置
# res = 256  # 圖像解析度 256x256

# # 創建模型
# model = UNet(in_channels, out_channels, num_additional_inputs, base_channels, res)

# # 打印每個層的大小
# for i, layer in enumerate(model.bottle_layers):
#     print(f"Layer {i}: {layer}")
#     # for name, param in layer.named_parameters():
#     #     print(f"  {name}: {param.size()}")

