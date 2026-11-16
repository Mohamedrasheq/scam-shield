"""
Split the cleaned data into train / validation / test.

Rules:
  - Validation and test come ONLY from real messages (imc2025, mishra_soni).
  - Synthetic messages (synthetic, synthetic_v2) go ONLY into training.
  - Real messages: 80% train, 10% validation, 10% test, same scam/safe mix in each.
  - Fixed seed, so the split is the same every run.
  - FROZEN val/test: if data/splits/val.csv and test.csv already exist, they are kept
    (minus rows the cleaning removed), and all new data goes to train. This keeps scores
    comparable between runs. Use --fresh to make a brand-new split.

Input : data/processed/clean.csv
Output: data/splits/train.csv, data/splits/val.csv, data/splits/test.csv

Run: .venv/bin/python split_data.py           (keep val/test)
     .venv/bin/python split_data.py --fresh   (new random split)
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
OUT = Path("data/splits")

df = pd.read_csv("data/processed/clean.csv", keep_default_na=False)
is_synthetic = df["source"].str.startswith("synthetic")
real = df[~is_synthetic]
synthetic = df[is_synthetic]
print(f"Real: {len(real)} | Synthetic: {len(synthetic)}")

def norm(texts):
    return texts.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


frozen = (OUT / "val.csv").exists() and (OUT / "test.csv").exists() and "--fresh" not in sys.argv
if frozen:
    # Keep the same val/test as before, so new runs are judged on the same messages
    old_val = set(norm(pd.read_csv(OUT / "val.csv", keep_default_na=False)["text"]))
    old_test = set(norm(pd.read_csv(OUT / "test.csv", keep_default_na=False)["text"]))
    key = norm(real["text"])
    val, test = real[key.isin(old_val)], real[key.isin(old_test)]
    real_train = real[~key.isin(old_val | old_test)]
    print(f"Frozen val/test kept: val {len(val)} (was {len(old_val)}), test {len(test)} (was {len(old_test)})")
else:
    # Stratify by label + source, so each split has the same mix
    strata = real["label"] + "_" + real["source"]
    real_train, rest = train_test_split(real, test_size=0.2, stratify=strata, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"] + "_" + rest["source"], random_state=SEED)

train = pd.concat([real_train, synthetic]).sample(frac=1, random_state=SEED)  # shuffle

# Safety check: no message text appears in two splits
def keys(frame):
    return set(norm(frame["text"]))

overlap = {"train-val": len(keys(train) & keys(val)), "train-test": len(keys(train) & keys(test)),
           "val-test": len(keys(val) & keys(test))}
print("Overlap between splits:", overlap)
assert sum(overlap.values()) == 0, "Same message found in two splits!"

OUT.mkdir(parents=True, exist_ok=True)
for name, frame in [("train", train), ("val", val), ("test", test)]:
    frame.to_csv(OUT / f"{name}.csv", index=False)

print("\nRows per split:")
summary = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
print(summary.groupby(["split", "source", "label"]).size().unstack(["label"]).fillna(0).astype(int).to_string())
print("\nTotals:")
print(summary.groupby(["split", "label"]).size().unstack().to_string())
print(f"\nSaved to {OUT}/")
