
from pathlib import Path
import pandas as pd

data_dir = Path("backend/data/processed/shenzhen")

train = pd.read_csv(data_dir / "train.csv")
validation = pd.read_csv(data_dir / "validation.csv")
test = pd.read_csv(data_dir / "test.csv")

splits = {
    "train": train,
    "validation": validation,
    "test": test,
}

# Check that every image belongs to only one split
image_sets = {
    name: set(df["image"])
    for name, df in splits.items()
}

assert image_sets["train"].isdisjoint(image_sets["validation"])
assert image_sets["train"].isdisjoint(image_sets["test"])
assert image_sets["validation"].isdisjoint(image_sets["test"])

# Check total image count
total = sum(len(df) for df in splits.values())
assert total == 662, f"Expected 662 rows, found {total}"

# Check labels
for name, df in splits.items():
    assert df["label"].notna().all(), f"Missing labels in {name}"
    assert set(df["label"].unique()).issubset({0, 1})

print("SUCCESS: No overlapping images.")
print(f"Total images across all splits: {total}")

for name, df in splits.items():
    print(f"{name}: {len(df)} images")
