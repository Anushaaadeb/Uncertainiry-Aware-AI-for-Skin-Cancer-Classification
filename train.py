import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
from pathlib import Path

from model import DualModeClassifier
from labels import DX_CODE_TO_LABEL, DX_CODES, HAM_COUNTS, NUM_CLASSES
from config import IMG_SIZE

SEED = 42
NUM_EPOCHS = 10  # was 3 -- 3 epochs is not enough for a fine-tune to converge;
                  # raise further for production, this is still a light default.


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Focal Loss to handle extreme HAM10000 class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class HAM10000Dataset(Dataset):
    """
    Loads HAM10000 images + labels. Returns raw PIL images (label map
    only, no transform applied here) so that train/test splits can each
    apply their own transform via TransformWrapper below -- test data
    should not get random-flip/rotation augmentation.
    """
    def __init__(self, csv_file, img_dir):
        self.df = pd.read_csv(csv_file)

        # Fail loudly if the metadata contains a dx code labels.py
        # doesn't know about, instead of silently mislabeling it.
        unknown_codes = set(self.df['dx'].unique()) - set(DX_CODES)
        if unknown_codes:
            raise ValueError(
                f"Metadata contains dx codes not present in labels.DX_CODES: "
                f"{unknown_codes}. Update labels.py before training."
            )

        # Fixed label mapping from labels.py -- NOT sorted() at runtime.
        # This is what keeps training labels and inference display names
        # in agreement.
        self.label_map = DX_CODE_TO_LABEL
        self.df['label'] = self.df['dx'].map(self.label_map)

        # Recursively scan all subdirectories for any image extension
        img_path_obj = Path(img_dir)
        valid_exts = {'.jpg', '.jpeg', '.png'}

        all_images = [
            p for p in img_path_obj.rglob('*')
            if p.suffix.lower() in valid_exts
        ]

        # Build map linking image_id (stem) -> absolute file path string
        self.image_path_map = {p.stem: str(p) for p in all_images}

        print(f"Dataset Initialized: Found {len(self.image_path_map)} image files in '{img_dir}'.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['image_id']

        if img_id not in self.image_path_map:
            raise KeyError(f"Image ID '{img_id}' not found in image_path_map! Check folder structure.")

        img_path = self.image_path_map[img_id]
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx]['label']

        return image, label


class TransformWrapper(Dataset):
    """Applies a transform to a (possibly Subset-wrapped) base dataset.
    Lets train/test splits share one HAM10000Dataset instance (avoiding
    a second expensive rglob scan) while using different transforms."""
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def run_training():
    set_seed(SEED)

    if not os.path.exists("ham10000"):
        print("Downloading HAM10000 dataset...")
        os.system("kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p ham10000 --unzip")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    base_dataset = HAM10000Dataset("ham10000/HAM10000_metadata.csv", "ham10000")

    # Stratified 80/20 split -- plain random_split can starve the rare
    # classes (df=115, vasc=142 images total) from the test set, making
    # their precision/recall undefined or wildly noisy.
    labels_all = base_dataset.df['label'].values
    indices = np.arange(len(base_dataset))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=SEED, stratify=labels_all
    )

    train_dataset = TransformWrapper(Subset(base_dataset, train_idx), train_transform)
    test_dataset = TransformWrapper(Subset(base_dataset, test_idx), test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # HAM10000 class weighting -- counts come from labels.py, indexed by
    # dx code rather than a hand-written array, so this can't drift out
    # of sync with the actual label order again.
    total = sum(HAM_COUNTS)
    class_weights = torch.tensor(
        [total / (NUM_CLASSES * c) for c in HAM_COUNTS]
    ).to(device)

    model = DualModeClassifier(num_classes=NUM_CLASSES, dropout_rate=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    best_val_acc = 0.0
    print(f"Training model on {device} using Focal Loss for {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | train_loss: {train_loss:.4f} "
              f"| val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} "
              f"| lr: {scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "fine_tuned_model.pth")
            print(f"  -> New best val_acc {val_acc:.4f}, saved to 'fine_tuned_model.pth'")

    print(f"Training complete! Best val_acc: {best_val_acc:.4f}. "
          f"Best weights saved to 'fine_tuned_model.pth'")

    return test_loader


if __name__ == "__main__":
    run_training()