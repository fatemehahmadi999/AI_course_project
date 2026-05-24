from dataset import HistologyDataset
import matplotlib.pyplot as plt

IMAGE_DIR = "/Users/fatemehahmadi/Projects/AI course/Project/Project_self/data/exported_dataset"

MASK_DIR = IMAGE_DIR

dataset = HistologyDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR
)

print("Dataset size:", len(dataset))

image, mask = dataset[0]

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(image.permute(1,2,0))
plt.title("Image")

plt.subplot(1,2,2)
plt.imshow(mask)
plt.title("Mask")

plt.show()