import torch
from model import UNet

model = UNet(in_channels=3, num_classes=4)

x = torch.randn(2, 3, 512, 512)

y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)