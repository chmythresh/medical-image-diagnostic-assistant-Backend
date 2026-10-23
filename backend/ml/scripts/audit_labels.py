
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "shenzhen"
LABELS_CSV = (
    PROJECT_ROOT / "data" / "processed" / "shenzhen"
    / "shenzhen_labels.csv"
)
OUTPUT_CSV = (
    PROJECT_ROOT / "artifacts" / "label_audit.csv"
)

df = pd.read_csv(LABELS_CSV)
records = []

for _, row in df.iterrows():
    image_name = str(row["image"])
    image_path = DATA_DIR / image_name
    reading_path = (
        DATA_DIR / "ClinicalReadings"
        / f"{Path(image_name).stem}.txt"
    )

    if reading_path.exists():
        reading = reading_path.read_text(
            encoding="utf-8", errors="replace"
        ).strip()
    else:
        reading = ""

    suffix = Path(image_name).stem[-1]
    expected_label = 0 if suffix == "0" else 1

    records.append({
        "image": image_name,
        "csv_label": int(row["label"]),
        "expected_label_from_suffix": expected_label,
        "label_matches_suffix": (
            int(row["label"]) == expected_label
        ),
        "image_exists": image_path.exists(),
        "reading_exists": reading_path.exists(),
        "clinical_reading": reading.replace("\n", " | "),
    })

audit = pd.DataFrame(records)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
audit.to_csv(OUTPUT_CSV, index=False)

print("Label audit completed.")
print(f"Images checked: {len(audit)}")
print(
    "CSV labels matching filename suffix:",
    int(audit["label_matches_suffix"].sum()),
)
print("Missing images:", int((~audit["image_exists"]).sum()))
print("Missing readings:", int((~audit["reading_exists"]).sum()))
print(f"Saved report to: {OUTPUT_CSV}")
print("\nSample clinical readings:")
print(audit[["image", "csv_label", "clinical_reading"]].head(10).to_string(index=False))
