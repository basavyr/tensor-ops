from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import os


def extract_training_images(dataset):
    # Extract training images
    print("Extracting training images...")
    train_dir = output_dir / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(tqdm(dataset['train'], desc="Training images")):
        try:
            label = item['label']
            img = item['image']

            # Convert to RGB if necessary (handles RGBA, L, P, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Create class folder (using label as folder name)
            class_dir = train_dir / f"{label:04d}"  # Format: 0000, 0001, etc.
            class_dir.mkdir(exist_ok=True)

            # Save image
            img.save(class_dir / f"{idx:07d}.JPEG")
        except Exception as e:
            print(f"\nError processing training image {idx}: {e}")
            continue


def extract_validation_images(dataset):
    # Extract validation images
    print("Extracting validation images...")
    val_dir = output_dir / 'val'
    val_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(tqdm(dataset['validation'], desc="Validation images")):
        try:
            label = item['label']
            img = item['image']

            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Create class folder
            class_dir = val_dir / f"{label:04d}"
            class_dir.mkdir(exist_ok=True)

            # Save image
            img.save(class_dir / f"{idx:07d}.JPEG")
        except Exception as e:
            print(f"\nError processing validation image {idx}: {e}")
            continue


def load_imagenet_from_folder(imagenet_folder: str):
    bs = 256
    from torchvision.datasets import ImageFolder
    from torchvision import transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(
        f'{imagenet_folder}/train', transform=transform)
    val_dataset = ImageFolder(f'{imagenet_folder}/val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=12)
    test_loader = DataLoader(val_dataset, batch_size=bs)


if __name__ == "__main__":
    # Define paths
    local_cache_dir = '/home/robertp/.cache/huggingface/_data/imagenet-1k/data'
    dataset_cache_dir = '/home/robertp/.cache/huggingface/_data/imagenet-1k/processed_cache'
    # Load from your parquet files
    # dataset = load_dataset(
    #     "parquet",
    #     data_files={
    #         'train': f'{local_cache_dir}/train-*.parquet',
    #         'validation': f'{local_cache_dir}/validation-*.parquet'
    #     },
    #     cache_dir=dataset_cache_dir,
    # )
    # extract_training_images(dataset)
    # extract_validation_images(dataset)

    imagenet_folder = "/home/robertp/.cache/huggingface/_data/imagenet_extracted"
    load_imagenet_from_folder(imagenet_folder)
