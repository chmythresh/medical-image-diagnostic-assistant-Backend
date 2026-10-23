
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

csv_path = Path("backend/data/processed/shenzhen/shenzhen_labels.csv")
output_dir = Path("backend/data/processed/shenzhen")
df = pd.read_csv(csv_path)

# Split into 70% training and 30% temporary data
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42,
)

# Split the remaining data into validation and test sets
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42,
)

train_df.to_csv(output_dir / "train.csv", index=False)
val_df.to_csv(output_dir / "validation.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)

print("Dataset split completed!")
print(f"Training:   {len(train_df)} images")
print(f"Validation: {len(val_df)} images")
print(f"Test:       {len(test_df)} images")

for name, split in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Test", test_df),
]:
    print(f"\n{name} class counts:")
    print(split["class_name"].value_counts().to_string())

print("\nCSV files saved successfully.")
