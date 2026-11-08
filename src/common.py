from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

LABEL_CANDIDATES = ["account_type", "label", "bot", "class", "is_bot", "target"]
ID_COLUMNS = {"id", "id_str", "user_id", "userid", "account_id", "screen_name_id"}
TEXT_CANDIDATES = ["screen_name", "name", "description", "status", "text", "tweet", "tweets", "location", "url", "lang"]
NUMERIC_CANDIDATES = [
    "followers_count", "followers", "friends_count", "following_count", "following",
    "listed_count", "favourites_count", "favorites_count", "statuses_count", "tweet_count",
    "account_age_days", "verified", "default_profile", "default_profile_image", "has_extended_profile"
]
CATEGORICAL_CANDIDATES = ["verified", "lang", "default_profile", "default_profile_image", "has_extended_profile"]


def read_csv_safely(path):
    """Read a CSV using common encodings and delimiter inference."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}\nPut the dataset in data/raw/ or pass --csv with its path.")
    errors = []
    for encoding in ("utf-8", "utf-8-sig", "ISO-8859-1"):
        try:
            return pd.read_csv(path, encoding=encoding, sep=None, engine="python")
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("Could not read the CSV. Attempts:\n" + "\n".join(errors))


def find_label_column(df):
    by_lower = {str(c).strip().lower(): c for c in df.columns}
    for candidate in LABEL_CANDIDATES:
        if candidate in by_lower:
            return by_lower[candidate]
    return None


def normalize_labels(series):
    """Convert common bot/human labels to 1=bot, 0=human; reject ambiguous labels."""
    s = series.copy()
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int), {"False": 0, "True": 1}
    if pd.api.types.is_numeric_dtype(s):
        vals = set(pd.to_numeric(s, errors="coerce").dropna().unique().tolist())
        if vals.issubset({0, 1}):
            return pd.to_numeric(s, errors="coerce").astype("Int64").astype(int), {"0": 0, "1": 1}
        # Common numeric labels like 1=human, 2=bot are not guessed.
        raise ValueError(f"Numeric labels are {sorted(vals)}. Please define their mapping explicitly; only 0/1 is auto-mapped.")
    cleaned = s.astype("string").str.strip().str.lower()
    mapping = {}
    for value in cleaned.dropna().unique():
        if value in {"bot", "bots", "social bot", "automated", "automation", "1", "true", "yes"}:
            mapping[value] = 1
        elif value in {"human", "real", "non-bot", "nonbot", "non bot", "genuine", "0", "false", "no"}:
            mapping[value] = 0
    unknown = sorted(set(cleaned.dropna().unique()) - set(mapping))
    if unknown:
        raise ValueError(f"Unrecognized label values: {unknown[:20]}. Update normalize_labels() with the correct label mapping.")
    result = cleaned.map(mapping)
    if result.isna().any():
        raise ValueError("The label column contains missing values. Clean or remove those rows first.")
    return result.astype(int), mapping
