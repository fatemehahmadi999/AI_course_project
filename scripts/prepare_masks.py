from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# =====================
# PATH TO ONE MASK
# =====================

mask_path = "/Users/fatemehahmadi/Projects/AI course/Project/Project_self/data/exported_dataset/PDX_H5471_ID_511_LG_HE [x=6656,y=215040,w=512,h=512]-labelled.png"

# =====================
# LOAD MASK
# =====================

mask = Image.open(mask_path)
mask_np = np.array(mask)

print("Mask shape:", mask_np.shape)

# =====================
# SHOW MASK
# =====================

plt.imshow(mask_np)
plt.title("Original RGB Mask")
plt.axis("off")
plt.show()

# =====================
# PRINT UNIQUE COLORS
# =====================

unique_colors = np.unique(mask_np.reshape(-1, mask_np.shape[-1]), axis=0)

print("\nUnique RGB colors found:\n")

for color in unique_colors:
    print(color)