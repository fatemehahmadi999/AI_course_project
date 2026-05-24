import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import HistologyDataset
from model import UNet


# =====================
# PATHS
# =====================

DATA_DIR = "/Users/fatemehahmadi/Projects/AI course/Project/Project_self/data/exported_dataset"

MODEL_PATH = "/Users/fatemehahmadi/Projects/AI course/Project/Project_self/models/unet_histology.pth"


# =====================
# DEVICE
# =====================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# =====================
# LOAD DATASET
# =====================

dataset = HistologyDataset(
    image_dir=DATA_DIR,
    mask_dir=DATA_DIR
)

image, true_mask = dataset[0]

image_input = image.unsqueeze(0).to(device)


# =====================
# LOAD MODEL
# =====================

model = UNet(in_channels=3, num_classes=4)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.to(device)
model.eval()


# =====================
# PREDICTION
# =====================

with torch.no_grad():

    output = model(image_input)

    prediction = torch.argmax(output, dim=1)

prediction = prediction.squeeze().cpu().numpy()


# =====================
# VISUALIZATION
# =====================

image_np = image.permute(1,2,0).numpy()

fig, axes = plt.subplots(1, 3, figsize=(15,5))

axes[0].imshow(image_np)
axes[0].set_title("H&E Image")
axes[0].axis("off")

axes[1].imshow(true_mask)
axes[1].set_title("Ground Truth")
axes[1].axis("off")

axes[2].imshow(prediction)
axes[2].set_title("Prediction")
axes[2].axis("off")

plt.tight_layout()
plt.show()
