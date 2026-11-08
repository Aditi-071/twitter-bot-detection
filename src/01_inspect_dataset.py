import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RESULTS_DIR, read_csv_safely, find_label_column, ID_COLUMNS, normalize_labels


def main():
    parser = argparse.ArgumentParser(description="Inspect a social-media bot dataset CSV")
    parser.add_argument("--csv", default=str(ROOT / "data" / "raw" / "twitter_bots.csv"), help="Path to input CSV")
    args = parser.parse_args()
    df = read_csv_safely(args.csv)

    lines = []
    lines.append(f"File: {Path(args.csv).resolve()}")
    lines.append(f"Shape: {df.shape}")
    lines.append("\nColumns and data types:")
    lines.extend([f"  {col}: {dtype}" for col, dtype in df.dtypes.items()])
    lines.append("\nMissing values:")
    lines.extend([f"  {col}: {int(n)}" for col, n in df.isna().sum().items()])
    lines.append(f"\nTotal missing cells: {int(df.isna().sum().sum())}")
    lines.append(f"Duplicate rows: {int(df.duplicated().sum())}")
    lines.append(f"Duplicate values in likely ID columns:")
    found_ids = [c for c in df.columns if str(c).strip().lower() in ID_COLUMNS]
    if found_ids:
        lines.extend([f"  {c}: {int(df[c].duplicated().sum())}" for c in found_ids])
    else:
        lines.append("  No recognized ID column found")

    label_col = find_label_column(df)
    lines.append(f"\nDetected label column: {label_col}")
    if label_col:
        lines.append("Label distribution (original values):")
        lines.extend([f"  {k}: {v}" for k, v in df[label_col].value_counts(dropna=False).items()])
        try:
            y, mapping = normalize_labels(df[label_col])
            lines.append(f"Binary mapping (1=bot, 0=human): {mapping}")
            lines.append(f"Normalized bot count: {int((y == 1).sum())}")
            lines.append(f"Normalized human count: {int((y == 0).sum())}")
        except Exception as exc:
            lines.append(f"Label mapping check: {exc}")
    else:
        lines.append("No label column found. Expected one of: account_type, label, bot, class, is_bot, target.")

    excluded = {str(c).strip().lower() for c in df.columns if str(c).strip().lower() in ID_COLUMNS}
    if label_col:
        excluded.add(str(label_col).strip().lower())
    candidate_features = [c for c in df.columns if str(c).strip().lower() not in excluded]
    lines.append(f"\nPotential non-ID feature columns: {candidate_features}")
    if not candidate_features:
        lines.append("WARNING: There are no predictive feature columns besides the ID and label. Do not train a classifier on ID alone.")
    elif all(str(c).strip().lower() in ID_COLUMNS for c in candidate_features):
        lines.append("WARNING: Only identifier columns remain. A meaningful predictive model cannot be trained.")

    print("\n".join(lines))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "dataset_report.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to: {out}")
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))

if __name__ == "__main__":
    main()
