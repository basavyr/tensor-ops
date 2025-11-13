from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import os


local_cache_dir = '/home/robertp/.cache/huggingface/_data/imagenet-1k/data'
# Load from your parquet files
dataset = load_dataset(
    "parquet",
    data_files={
        'train': f'{local_cache_dir}/train-*.parquet',
        'validation': f'{local_cache_dir}/validation-*.parquet'
    }
)

# Define output directory
output_dir = Path("/home/robertp/.cache/huggingface/_data/imagenet_extracted")

# Extract training images
print("Extracting training images...")
train_dir = output_dir / 'train'
train_dir.mkdir(parents=True, exist_ok=True)

for idx, item in enumerate(tqdm(dataset['train'], desc="Training images")):
    label = item['label']
    img = item['image']

    # Create class folder (using label as folder name)
    class_dir = train_dir / f"{label:04d}"  # Format: 0000, 0001, etc.
    class_dir.mkdir(exist_ok=True)

    # Save image
    img.save(class_dir / f"{idx:07d}.JPEG")

# Extract validation images
print("Extracting validation images...")
val_dir = output_dir / 'val'
val_dir.mkdir(parents=True, exist_ok=True)

for idx, item in enumerate(tqdm(dataset['validation'], desc="Validation images")):
    label = item['label']
    img = item['image']

    # Create class folder
    class_dir = val_dir / f"{label:04d}"
    class_dir.mkdir(exist_ok=True)

    # Save image
    img.save(class_dir / f"{idx:07d}.JPEG")

print(f"Extraction complete! Images saved to {output_dir}")
