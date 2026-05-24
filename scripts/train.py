from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import HistologyDataset
from model import UNet


DATA_DIR = "/content/drive/MyDrive/histology_project/exported_dataset"
MODEL_DIR = Path("/Users/fatemehahmadi/Projects/AI course/Project/Project_self/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 2
NUM_CLASSES = 4
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


dataset = HistologyDataset(
    image_dir=DATA_DIR,
    mask_dir=DATA_DIR
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)

class_weights = torch.tensor(
    [0.2, 1.0, 1.0, 1.0],
    dtype=torch.float32
).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")


save_path = MODEL_DIR / "unet_histology.pth"
torch.save(model.state_dict(), save_path)

print("Model saved to:", save_path)