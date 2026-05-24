from pathlib import Path
from PIL import Image
import numpy as np
from collections import Counter

MASK_DIR = Path("/Users/fatemehahmadi/Projects/AI course/Project/Project_self/data/exported_dataset")

mask_files = sorted(MASK_DIR.glob("*-labelled.png"))

print(f"Found {len(mask_files)} mask files")

all_values = Counter()

for mask_path in mask_files:
    mask = np.array(Image.open(mask_path))
    values, counts = np.unique(mask, return_counts=True)

    for v, c in zip(values, counts):
        all_values[int(v)] += int(c)

print("Label values across all masks:")
for label, count in sorted(all_values.items()):
    print(f"Label {label}: {count} pixels")