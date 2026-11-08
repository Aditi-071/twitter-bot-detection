# Social Media Bot Detection — VS Code Project

This is a cleaned-up, beginner-friendly replacement for the old Twitter Bot Detection code.

## Important dataset limitation

The Kaggle file you described has **37,438 rows and only two columns: `id` and `account_type`**. This is enough to inspect labels and count bots/humans, but **not enough to train a meaningful predictive model**: `id` is an identifier, not a behavioral signal. The training script will detect this and stop with a clear explanation rather than report misleading accuracy.

The model pipeline also supports a feature-rich CSV with a bot/human label and account/text fields, such as the older project's `training_data_2_csv_UTF.csv` schema.

## What was changed

- Removed the old live Twitter/X API collection code and embedded credentials.
- Removed the obsolete `api.list_members` usage.
- Replaced deprecated pandas `.ix` usage and brittle hard-coded paths.
- Added dataset inspection and label-distribution reporting.
- Added a guarded ML pipeline using text TF-IDF plus available numeric/categorical account features.
- Added train/test split, metrics, confusion matrix, classification report, and saved model.
- Added an optional Streamlit dataset-inspection interface.

## 1. Open in VS Code

Extract this ZIP and open the `bot_detection_vscode_fixed` folder in VS Code.

## 2. Create a virtual environment (Windows PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

## 3. Put your Kaggle CSV in `data/raw/`

For example:

```text
data/raw/twitter_bots.csv
```

Do not rename the columns unless you also change the code. Supported label columns include `account_type`, `label`, `bot`, and `class`.

## 4. Inspect the dataset

```powershell
python src/01_inspect_dataset.py --csv data/raw/twitter_bots.csv
```

This prints shape, columns, data types, missing values, duplicate rows, label counts, and whether there are usable predictive features. It also saves a text report in `results/`.

## 5. Train a model (only when features exist)

```powershell
python src/02_train_model.py --csv data/raw/twitter_bots.csv
```

If the file has only `id` and `account_type`, training intentionally stops. To train a model, use a dataset with actual features such as account profile fields, tweet text, follower/following counts, status counts, verification, or behavior. Do not use account ID as a predictive feature.

For a feature-rich CSV with fields like `screen_name`, `description`, `status`, `followers_count`, `friends_count`, `verified`, and `bot`, the pipeline combines available text and structured fields automatically.

## 6. View the optional app

```powershell
streamlit run app.py
```

Upload a CSV to inspect it. The app does not pretend it can predict from an ID-only dataset.

## Output files

- `results/dataset_report.txt`
- `results/model_metrics.json`
- `results/classification_report.txt`
- `results/confusion_matrix.png`
- `models/bot_detection_pipeline.joblib`

These files are generated only when the relevant step can run.

## Security

The original uploaded project contained hard-coded X/Twitter credentials. They have deliberately **not** been copied into this cleaned project. Because those credentials were shared in source code, revoke/rotate them in the X Developer Portal. This project does not require X API credentials to analyze a downloaded CSV.
