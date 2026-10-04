"""Clean the raw symptoms dataset and write data/medical_dataset_clean.csv.

Usage:
    python -m ml.clean_dataset [raw_csv] [output_csv]

The raw file (columns: Symptoms, Disease) is not versioned in this repository.
"""
import os
import re
import sys

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(ROOT_DIR, "data", "neurological_dataset_5000.csv")
DEFAULT_OUTPUT = os.path.join(ROOT_DIR, "data", "medical_dataset_clean.csv")


def clean_symptoms(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = [w for w in text.split() if len(w) > 2]

    seen = set()
    unique = []
    for w in words:
        if w not in seen:
            unique.append(w)
            seen.add(w)

    return ", ".join(unique)


def main(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT):
    df = pd.read_csv(input_path)
    df["Symptoms"] = df["Symptoms"].apply(clean_symptoms)
    df = df[df["Symptoms"].str.len() > 0]
    df.to_csv(output_path, index=False)
    print(f"Clean dataset written to {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
