"""
Brain Tumor MRI Classification — Evaluation Script
====================================================
Loads a trained model and generates comprehensive evaluation metrics.

Usage:
    python evaluate.py --data_dir ./dataset --model_path ./model/brain_tumor_classifier.pth
"""

import os
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
import numpy as np

# Try to import matplotlib for visualizations
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("⚠️  matplotlib/seaborn not installed. Skipping plot generation.")
    print("   Install with: pip install matplotlib seaborn")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def build_model(num_classes=4):
    """Reconstruct EfficientNet-B0 architecture (without pretrained weights)."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )
    return model


def plot_confusion_matrix(cm, class_names, output_path):
    """Generate a beautiful confusion matrix heatmap."""
    if not HAS_PLOT:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title("Confusion Matrix — Brain Tumor Classification",
                 fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Predicted Label", fontsize=13, labelpad=10)
    ax.set_ylabel("True Label", fontsize=13, labelpad=10)
    ax.tick_params(axis="both", labelsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Confusion matrix saved to: {output_path}")


def plot_per_class_metrics(report_dict, class_names, output_path):
    """Bar chart comparing precision, recall, and F1 per class."""
    if not HAS_PLOT:
        return

    precisions = [report_dict[c]["precision"] for c in class_names]
    recalls    = [report_dict[c]["recall"]    for c in class_names]
    f1s        = [report_dict[c]["f1-score"]  for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, precisions, width, label="Precision",
                   color="#3b82f6", alpha=0.85)
    bars2 = ax.bar(x, recalls, width, label="Recall",
                   color="#06d6a0", alpha=0.85)
    bars3 = ax.bar(x + width, f1s, width, label="F1-Score",
                   color="#f59e0b", alpha=0.85)

    ax.set_xlabel("Tumor Category", fontsize=13, labelpad=10)
    ax.set_ylabel("Score", fontsize=13, labelpad=10)
    ax.set_title("Per-Class Classification Metrics", fontsize=16,
                 fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() for n in class_names], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Per-class metrics saved to: {output_path}")


def plot_training_history(history_path, output_path):
    """Plot training/validation loss and accuracy curves."""
    if not HAS_PLOT:
        return

    if not os.path.exists(history_path):
        print(f"   ⚠️  Training history not found at: {history_path}")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], "o-", color="#3b82f6",
             label="Train Loss", linewidth=2, markersize=4)
    ax1.plot(epochs, history["val_loss"], "o-", color="#ef4444",
             label="Val Loss", linewidth=2, markersize=4)
    ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, history["train_acc"], "o-", color="#3b82f6",
             label="Train Acc", linewidth=2, markersize=4)
    ax2.plot(epochs, history["val_acc"], "o-", color="#06d6a0",
             label="Val Acc", linewidth=2, markersize=4)
    ax2.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    plt.suptitle("Brain Tumor Classifier — Training History",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Training history plot saved to: {output_path}")


def main(args):
    print("=" * 60)
    print("  Brain Tumor MRI Classification — Evaluation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥  Device: {device}")

    # ── Load model ──
    if not os.path.exists(args.model_path):
        print(f"\n❌ Model file not found: {args.model_path}")
        print("   Please train the model first: python train.py")
        return

    model = build_model(num_classes=4)
    model.load_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()
    print(f"\n✅ Model loaded from: {args.model_path}")

    # ── Load test data ──
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    test_dir = os.path.join(args.data_dir, "Testing")
    if not os.path.isdir(test_dir):
        print(f"\n❌ Test directory not found: {test_dir}")
        return

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=2
    )
    class_names = test_dataset.classes
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Classes: {class_names}")

    # ── Run inference ──
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # ── Metrics ──
    print("\n" + "=" * 60)
    print("  Classification Report")
    print("=" * 60)

    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4
    )
    print(f"\n{report}")

    report_dict = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )

    accuracy  = accuracy_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall    = recall_score(all_labels, all_preds, average="weighted")

    print(f"Overall Accuracy:    {accuracy:.4f}")
    print(f"Weighted F1 Score:   {f1:.4f}")
    print(f"Weighted Precision:  {precision:.4f}")
    print(f"Weighted Recall:     {recall:.4f}")

    # ── Confusion Matrix ──
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix:")
    print(cm)

    # ── Generate plots ──
    os.makedirs(args.output_dir, exist_ok=True)

    plot_confusion_matrix(
        cm, class_names,
        os.path.join(args.output_dir, "confusion_matrix.png")
    )

    plot_per_class_metrics(
        report_dict, class_names,
        os.path.join(args.output_dir, "per_class_metrics.png")
    )

    plot_training_history(
        os.path.join(os.path.dirname(args.model_path), "training_history.json"),
        os.path.join(args.output_dir, "training_history.png")
    )

    # ── Save results as JSON ──
    results = {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "per_class": {
            name: {
                "precision": float(report_dict[name]["precision"]),
                "recall": float(report_dict[name]["recall"]),
                "f1_score": float(report_dict[name]["f1-score"]),
                "support": int(report_dict[name]["support"]),
            }
            for name in class_names
        },
        "confusion_matrix": cm.tolist(),
    }

    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Evaluation results saved to: {results_path}")

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Brain Tumor Classifier")
    parser.add_argument("--data_dir", type=str, default="./dataset",
                        help="Path to dataset directory")
    parser.add_argument("--model_path", type=str,
                        default="./model/brain_tumor_classifier.pth",
                        help="Path to trained model weights")
    parser.add_argument("--output_dir", type=str, default="./evaluation",
                        help="Directory to save evaluation outputs")
    args = parser.parse_args()

    main(args)
