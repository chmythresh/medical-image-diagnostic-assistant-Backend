
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

ROOT = Path("backend/data")
DATA_DIR = ROOT / "processed/shenzhen"
IMAGE_DIR = ROOT / "raw/shenzhen"
ARTIFACT_DIR = Path("backend/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 128
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_NAMES = {0: "normal", 1: "abnormal"}


class ChestXrayDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = IMAGE_DIR / row["image"]

        with Image.open(path) as image:
            image = image.convert("L")
            image = self.transform(image)

        return image, int(row["label"]), row["image"]


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


test_dataset = ChestXrayDataset(DATA_DIR / "test.csv")
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

model = SmallCNN().to(DEVICE)
checkpoint = ARTIFACT_DIR / "best_cnn.pt"

model.load_state_dict(
    torch.load(checkpoint, map_location=DEVICE, weights_only=True)
)
model.eval()

records = []

with torch.no_grad():
    for images, labels, filenames in test_loader:
        outputs = model(images.to(DEVICE))
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        predictions = outputs.argmax(dim=1).cpu().tolist()

        for i, filename in enumerate(filenames):
            true_label = int(labels[i])
            predicted_label = int(predictions[i])

            records.append({
                "image": filename,
                "true_label": true_label,
                "true_class": LABEL_NAMES[true_label],
                "predicted_label": predicted_label,
                "predicted_class": LABEL_NAMES[predicted_label],
                "confidence": float(probabilities[i, predicted_label]),
                "correct": true_label == predicted_label,
            })

results = pd.DataFrame(records)
errors = results[~results["correct"]].copy()

results.to_csv(ARTIFACT_DIR / "test_predictions.csv", index=False)
errors.to_csv(ARTIFACT_DIR / "misclassified_test.csv", index=False)

# Save confusion matrix
cm = confusion_matrix(
    results["true_label"],
    results["predicted_label"],
    labels=[0, 1],
)

fig, ax = plt.subplots(figsize=(6, 5))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Abnormal"],
)
display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
ax.set_title("CNN Test Confusion Matrix")
fig.tight_layout()
fig.savefig(
    ARTIFACT_DIR / "confusion_matrix.png",
    dpi=160,
    bbox_inches="tight",
)
plt.close(fig)

# Save a gallery of incorrectly classified images
if len(errors) > 0:
    columns = 3
    rows = int(np.ceil(len(errors) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(12, 4 * rows), squeeze=False
    )

    for ax, (_, row) in zip(axes.flat, errors.iterrows()):
        path = IMAGE_DIR / row["image"]

        with Image.open(path) as image:
            ax.imshow(image.convert("L"), cmap="gray")

        ax.set_title(
            f"{row['image']}\n"
            f"True: {row['true_class']} | "
            f"Predicted: {row['predicted_class']}\n"
            f"Model confidence: {row['confidence']:.2f}",
            fontsize=9,
        )
        ax.axis("off")

    for ax in axes.flat[len(errors):]:
        ax.axis("off")

    fig.suptitle("Misclassified Test Images", fontsize=14)
    fig.tight_layout()
    fig.savefig(
        ARTIFACT_DIR / "misclassified_gallery.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

print("Visual evaluation completed.")
print(f"Test images: {len(results)}")
print(f"Correct predictions: {int(results['correct'].sum())}")
print(f"Incorrect predictions: {len(errors)}")
print("\nMisclassified image counts:")
print(
    errors.groupby(["true_class", "predicted_class"])
    .size()
    .to_string()
)
print(f"\nSaved outputs to: {ARTIFACT_DIR}")
print("- confusion_matrix.png")
print("- misclassified_gallery.png")
print("- misclassified_test.csv")
print("- test_predictions.csv")
