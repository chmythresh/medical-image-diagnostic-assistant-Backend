
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "backend" / "data"
DATA_DIR = DATA_ROOT / "processed" / "shenzhen"
IMAGE_DIR = DATA_ROOT / "raw" / "shenzhen"
ARTIFACT_DIR = PROJECT_ROOT / "backend" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.00001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device:", DEVICE)


class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = IMAGE_DIR / row["image"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return image, label


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

train_dataset = ChestXrayDataset(
    DATA_DIR / "train.csv", transform=transform
)
val_dataset = ChestXrayDataset(
    DATA_DIR / "validation.csv", transform=transform
)
test_dataset = ChestXrayDataset(
    DATA_DIR / "test.csv", transform=transform
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))

# Load ImageNet-pretrained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all layers first
for parameter in model.parameters():
    parameter.requires_grad = False

# Unfreeze the final residual block
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

# Replace the classifier for our two classes
number_of_features = model.fc.in_features
model.fc = nn.Linear(number_of_features, 2)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

# Use a lower learning rate for pretrained features
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": LEARNING_RATE},
    {"params": model.fc.parameters(), "lr": LEARNING_RATE * 10},
])

checkpoint_path = ARTIFACT_DIR / "best_resnet18_finetuned.pt"
best_val_loss = float("inf")

# Train and select checkpoint using validation loss
for epoch in range(EPOCHS):
    model.train()

    # Keep frozen layers in evaluation mode, including BatchNorm
    model.conv1.eval()
    model.bn1.eval()
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()

    train_loss_sum = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * labels.size(0)

    train_loss = train_loss_sum / len(train_dataset)

    model.eval()
    val_loss_sum = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss_sum += loss.item() * labels.size(0)

    val_loss = val_loss_sum / len(val_dataset)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), checkpoint_path)

print("\nBest checkpoint saved to:", checkpoint_path)

# Evaluate the selected checkpoint on the test set
model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )
)
model.eval()

true_labels = []
predicted_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu().tolist()

        predicted_labels.extend(predictions)
        true_labels.extend(labels.tolist())

metrics = {
    "model": "ResNet18 fine-tuned layer4 and classifier",
    "test_images": len(test_dataset),
    "accuracy": accuracy_score(true_labels, predicted_labels),
    "precision_macro": precision_score(
        true_labels, predicted_labels,
        average="macro", zero_division=0
    ),
    "recall_macro": recall_score(
        true_labels, predicted_labels,
        average="macro", zero_division=0
    ),
    "f1_macro": f1_score(
        true_labels, predicted_labels,
        average="macro", zero_division=0
    ),
    "confusion_matrix": confusion_matrix(
        true_labels, predicted_labels, labels=[0, 1]
    ).tolist(),
}

print("\nFine-tuned ResNet18 test metrics:")
print(json.dumps(metrics, indent=2))

print("\nClassification report:")
print(classification_report(
    true_labels,
    predicted_labels,
    labels=[0, 1],
    target_names=["normal", "abnormal"],
    zero_division=0,
))

metrics_path = ARTIFACT_DIR / "resnet18_finetuned_test_metrics.json"
with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print("Metrics saved to:", metrics_path)
