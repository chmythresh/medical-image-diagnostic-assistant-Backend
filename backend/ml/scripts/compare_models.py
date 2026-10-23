
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = PROJECT_ROOT / "backend" / "artifacts"

RESULT_FILES = {
    "Original CNN": "test_metrics.json",
    "Frozen ResNet18": "resnet18_test_metrics.json",
    "Fine-tuned ResNet18": "resnet18_finetuned_test_metrics.json",
}

records = []

for model_name, filename in RESULT_FILES.items():
    path = ARTIFACT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    records.append({
        "model": model_name,
        "test_images": metrics.get("test_images", 100),
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "confusion_matrix": json.dumps(
            metrics["confusion_matrix"]
        ),
    })

df = pd.DataFrame(records)

csv_path = ARTIFACT_DIR / "model_comparison.csv"
json_path = ARTIFACT_DIR / "model_comparison.json"

df.to_csv(csv_path, index=False)

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(records, file, indent=2)

print("\nModel comparison (exploratory):")
print(
    df.drop(columns=["confusion_matrix"])
    .to_string(index=False, formatters={
        "accuracy": "{:.3f}".format,
        "precision_macro": "{:.3f}".format,
        "recall_macro": "{:.3f}".format,
        "f1_macro": "{:.3f}".format,
    })
)

print(f"\nCSV report saved to: {csv_path}")
print(f"JSON report saved to: {json_path}")
