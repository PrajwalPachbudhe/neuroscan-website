"""
Brain Tumor MRI Classification — Training Script
=================================================
Fine-tunes EfficientNet-B0 on brain tumor MRI images.
Categories: Glioma, Meningioma, No Tumor, Pituitary

Usage:
    python train.py --data_dir ./dataset --epochs 20 --batch_size 32
"""

import os
import argparse
import time
import copy
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]  # ImageNet stats
STD  = [0.229, 0.224, 0.225]


def get_transforms():
    """Data augmentation for training, standard preprocessing for validation."""
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    return train_transform, test_transform


def build_model(num_classes=4, pretrained=True):
    """
    Build EfficientNet-B0 with a custom classification head.
    Freezes early layers for transfer learning.
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Freeze all feature extraction layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Unfreeze last 2 blocks for fine-tuning
    for param in model.features[-2:].parameters():
        param.requires_grad = True

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )

    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch, return loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 20 == 0:
            print(f"    Batch [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model, return loss, accuracy, and predictions."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def main(args):
    print("=" * 60)
    print("  Brain Tumor MRI Classification — Training")
    print("=" * 60)

    # ── Device setup ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # ── Data loading ──
    train_transform, test_transform = get_transforms()

    train_dir = os.path.join(args.data_dir, "Training")
    test_dir  = os.path.join(args.data_dir, "Testing")

    if not os.path.isdir(train_dir):
        print(f"\n❌ Training directory not found: {train_dir}")
        print("   Please download the dataset from Kaggle:")
        print("   https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=test_transform)

    print(f"\n📂 Dataset loaded:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Testing samples:  {len(test_dataset)}")
    print(f"   Classes: {train_dataset.classes}")

    # Verify class ordering matches our expected order
    class_to_idx = train_dataset.class_to_idx
    print(f"   Class mapping: {class_to_idx}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Model setup ──
    model = build_model(num_classes=len(train_dataset.classes))
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model: EfficientNet-B0")
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # ── Training setup ──
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training loop ──
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n🚀 Starting training for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, test_loader, criterion, device
        )

        # Step scheduler
        scheduler.step()

        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:02d}/{args.epochs}]  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%  │  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%  │  "
            f"LR: {lr:.6f}  Time: {elapsed:.1f}s"
        )

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"   ✅ New best model! Accuracy: {val_acc:.2f}%")

    # ── Save best model ──
    model.load_state_dict(best_model_wts)
    save_path = os.path.join(args.output_dir, "brain_tumor_classifier.pth")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n💾 Best model saved to: {save_path}")
    print(f"   Best validation accuracy: {best_acc:.2f}%")

    # Save class mapping for inference
    mapping_path = os.path.join(args.output_dir, "class_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"   Class mapping saved to: {mapping_path}")

    # Save training history
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"   Training history saved to: {history_path}")

    # ── Final evaluation ──
    print("\n" + "=" * 60)
    print("  Final Evaluation on Test Set")
    print("=" * 60)

    _, final_acc, preds, labels = validate(model, test_loader, criterion, device)

    # Classification report
    class_names = list(class_to_idx.keys())
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    print(f"\n{report}")

    # F1 score
    f1 = f1_score(labels, preds, average="weighted")
    print(f"Weighted F1 Score: {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print(f"\nConfusion Matrix:")
    print(cm)

    print("\n✅ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Brain Tumor Classifier")
    parser.add_argument("--data_dir", type=str, default="./dataset",
                        help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./model",
                        help="Directory to save trained model")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of data loading workers")
    args = parser.parse_args()

    main(args)
