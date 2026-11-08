import argparse
from pathlib import Path
import json
import sys
import warnings
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RESULTS_DIR, MODELS_DIR, read_csv_safely, find_label_column, ID_COLUMNS, TEXT_CANDIDATES, NUMERIC_CANDIDATES, CATEGORICAL_CANDIDATES, normalize_labels


def main():
    parser = argparse.ArgumentParser(description="Train a simple bot-vs-human classifier")
    parser.add_argument("--csv", default=str(ROOT / "data" / "raw" / "twitter_bots.csv"), help="Path to input CSV")
    args = parser.parse_args()
    df = read_csv_safely(args.csv)
    label_col = find_label_column(df)
    if not label_col:
        raise SystemExit("ERROR: No label column found. Expected account_type, label, bot, class, is_bot, or target.")

    try:
        y, mapping = normalize_labels(df[label_col])
    except Exception as exc:
        raise SystemExit(f"ERROR: Cannot map labels safely. {exc}")

    lower_to_original = {str(c).strip().lower(): c for c in df.columns}
    id_cols = [c for c in df.columns if str(c).strip().lower() in ID_COLUMNS]
    feature_cols = [c for c in df.columns if c != label_col and c not in id_cols]
    if not feature_cols:
        raise SystemExit(
            "NOT ENOUGH FEATURES: this CSV contains only an account ID and its label.\n"
            "It can be used for label counts and data inspection, but not for meaningful supervised bot detection.\n"
            "Do not train on ID: account IDs are identifiers and would produce misleading results.\n"
            "Use a dataset containing account/tweet features (e.g. description, status text, followers_count,\n"
            "friends_count, statuses_count, verified, or other behavior/profile fields)."
        )

    work = df[feature_cols].copy()
    work = work.replace({"None": pd.NA, "none": pd.NA, "": pd.NA})

    # Build a single text field from available text columns. Limit the input text to avoid huge memory use.
    text_cols = [c for c in feature_cols if str(c).strip().lower() in TEXT_CANDIDATES]
    if text_cols:
        work["__combined_text__"] = work[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.slice(0, 20000)
    else:
        work["__combined_text__"] = ""

    # Select known numerical and categorical fields when present; do not use arbitrary IDs.
    numeric_cols = []
    categorical_cols = []
    for c in feature_cols:
        lc = str(c).strip().lower()
        if c in text_cols:
            continue
        if lc in CATEGORICAL_CANDIDATES:
            categorical_cols.append(c)
        elif lc in NUMERIC_CANDIDATES or pd.api.types.is_numeric_dtype(work[c]):
            if pd.api.types.is_bool_dtype(work[c]):
                categorical_cols.append(c)
            else:
                numeric_cols.append(c)
        elif work[c].dtype == "object" and work[c].nunique(dropna=True) <= 30:
            categorical_cols.append(c)

    # Normalize selected structured columns safely using standard string/object types (avoids pandas.NA TypeError)
    for c in numeric_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    for c in categorical_cols:
        work[c] = work[c].fillna("missing").astype(str)

    has_text = work["__combined_text__"].fillna("").astype(str).str.strip().ne("").any()
    transformers = []
    if has_text:
        transformers.append(("text", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2, sublinear_tf=True), "__combined_text__"))
    if numeric_cols:
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler(with_mean=False))])
        transformers.append(("numeric", num_pipe, numeric_cols))
    if categorical_cols:
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
        transformers.append(("categorical", cat_pipe, categorical_cols))

    if not transformers:
        raise SystemExit(
            "NOT ENOUGH USABLE FEATURES: columns were found, but none could be safely converted to text, numeric, or categorical predictors.\n"
            "Inspect the columns and add real account/tweet features. Do not use ID as a feature."
        )

    X = work[["__combined_text__"] + numeric_cols + categorical_cols].copy()
    if X.shape[1] == 1 and not has_text:
        raise SystemExit("NOT ENOUGH FEATURES: all available feature fields are empty. Add actual account or tweet features.")

    # Drop rows whose labels are missing (normalize_labels normally rejects these, retained as defensive check).
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    if y.nunique() < 2:
        raise SystemExit("Training needs both bot and human examples; only one class was found.")
    if y.value_counts().min() < 2:
        raise SystemExit("Need at least two examples in each class for a stratified train/test split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    best_name, best_f1, best_pipeline = None, -1, None

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42)
    }
    for name, estimator in models.items():
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)
        pipeline = Pipeline([("features", preprocessor), ("model", estimator)])
        print(f"\nTraining {name}...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        scores = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision_bot": float(precision_score(y_test, pred, pos_label=1, zero_division=0)),
            "recall_bot": float(recall_score(y_test, pred, pos_label=1, zero_division=0)),
            "f1_bot": float(f1_score(y_test, pred, pos_label=1, zero_division=0)),
        }
        try:
            if hasattr(pipeline.named_steps["model"], "predict_proba"):
                bot_scores = pipeline.predict_proba(X_test)[:, 1]
            elif hasattr(pipeline.named_steps["model"], "decision_function"):
                bot_scores = pipeline.decision_function(X_test)
            else:
                bot_scores = None
            scores["roc_auc"] = float(roc_auc_score(y_test, bot_scores)) if bot_scores is not None else None
        except Exception:
            scores["roc_auc"] = None
        results[name] = scores
        print(f"{name}: {scores}")
        print(classification_report(y_test, pred, labels=[0, 1], target_names=["Human", "Bot"], zero_division=0))

        if scores["f1_bot"] > best_f1:
            best_f1, best_name, best_pipeline = scores["f1_bot"], name, pipeline
            best_pred = pred

    (RESULTS_DIR / "model_metrics.json").write_text(json.dumps({"label_column": label_col, "label_mapping": mapping, "feature_columns": feature_cols, "text_columns": text_cols, "numeric_columns": numeric_cols, "categorical_columns": categorical_cols, "models": results, "selected_model_by_bot_f1": best_name}, indent=2), encoding="utf-8")
    (RESULTS_DIR / "classification_report.txt").write_text(classification_report(y_test, best_pred, labels=[0, 1], target_names=["Human", "Bot"], zero_division=0), encoding="utf-8")
    ConfusionMatrixDisplay.from_predictions(y_test, best_pred, labels=[0, 1], display_labels=["Human", "Bot"], cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()
    joblib.dump(best_pipeline, MODELS_DIR / "bot_detection_pipeline.joblib")
    print(f"\nSelected model by Bot F1: {best_name}")
    print(f"Metrics saved: {RESULTS_DIR / 'model_metrics.json'}")
    print(f"Model saved: {MODELS_DIR / 'bot_detection_pipeline.joblib'}")

if __name__ == "__main__":
    main()