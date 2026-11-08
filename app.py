from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import json

# Paths setup
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_PATH = MODELS_DIR / "bot_detection_pipeline.joblib"
METRICS_PATH = RESULTS_DIR / "model_metrics.json"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"

st.set_page_config(page_title="Social Media Bot Detector", page_icon="🤖", layout="wide")
st.title("🤖 Social Media Bot Detection App")

# Navigation or tabs
tab1, tab2 = st.tabs(["🔍 Predict Account", "📊 Dataset & Model Inspector"])

with tab1:
    st.header("Live Bot Prediction & Explanation")
    st.write("Enter the profile details below to check if the account is predicted to be a **Human** or a **Bot** and understand why.")

    if not MODEL_PATH.exists():
        st.warning("⚠️ No trained model found yet! Please run `python src/02_train_model.py` first to generate the model.")
    else:
        # Load the trained pipeline
        pipeline = joblib.load(MODEL_PATH)

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                screen_name = st.text_input("Screen Name / Handle", placeholder="johndoe123")
                description = st.text_area("Profile Description / Bio", placeholder="Just a regular person tweeting about daily life and tech.")
                location = st.text_input("Location", placeholder="New York, USA")
                lang = st.text_input("Language (lang)", placeholder="en")
                created_at = st.text_input("Created At (Timestamp)", placeholder="2020-01-01 00:00:00")
                profile_background_image_url = st.text_input("Profile Background Image URL", placeholder="http://abs.twimg.com/images/themes/theme1/bg.png")
                profile_image_url = st.text_input("Profile Image URL", placeholder="http://pbs.twimg.com/profile_images/example.jpg")

            with col2:
                followers_count = st.number_input("Followers Count", min_value=0, value=150)
                friends_count = st.number_input("Friends (Following) Count", min_value=0, value=200)
                favourites_count = st.number_input("Favourites Count", min_value=0, value=500)
                statuses_count = st.number_input("Statuses / Tweet Count", min_value=0, value=300)
                account_age_days = st.number_input("Account Age (Days)", min_value=0, value=730)
                average_tweets_per_day = st.number_input("Average Tweets Per Day", min_value=0.0, value=1.5)
                
                verified = st.selectbox("Verified Account?", [False, True])
                default_profile = st.selectbox("Default Profile?", [False, True])
                default_profile_image = st.selectbox("Default Profile Image?", [False, True])
                geo_enabled = st.selectbox("Geo Enabled?", [False, True])

            submit_button = st.form_submit_button("Predict & Explain")

        if submit_button:
            input_data = pd.DataFrame([{
                "Unnamed: 0": 0,
                "created_at": created_at if created_at else "2020-01-01 00:00:00",
                "default_profile": default_profile,
                "default_profile_image": default_profile_image,
                "description": description if description else "",
                "favourites_count": favourites_count,
                "followers_count": followers_count,
                "friends_count": friends_count,
                "geo_enabled": geo_enabled,
                "id": 123456789,
                "lang": lang if lang else "en",
                "location": location if location else "",
                "profile_background_image_url": profile_background_image_url if profile_background_image_url else "",
                "profile_image_url": profile_image_url if profile_image_url else "",
                "screen_name": screen_name if screen_name else "user",
                "statuses_count": statuses_count,
                "verified": verified,
                "average_tweets_per_day": average_tweets_per_day,
                "account_age_days": account_age_days
            }])

            text_cols = ["screen_name", "name", "description", "status", "text", "tweet", "tweets", "location", "url", "lang"]
            available_text_cols = [c for c in text_cols if c in input_data.columns]
            input_data["__combined_text__"] = (
                input_data[available_text_cols]
                .fillna("")
                .astype(str)
                .agg(" ".join, axis=1)
                .str.slice(0, 20000)
            )

            try:
                prediction = pipeline.predict(input_data)[0]
                
                prob_str = ""
                if hasattr(pipeline, "predict_proba"):
                    probs = pipeline.predict_proba(input_data)[0]
                    bot_prob = probs[1] * 100
                    prob_str = f" (Bot Probability: **{bot_prob:.1f}%**)"

                st.markdown("---")
                if prediction == 1:
                    st.error(f"🚨 **Prediction: BOT**{prob_str}")
                else:
                    st.success(f"✅ **Prediction: HUMAN**{prob_str}")

                st.subheader("💡 Why this prediction?")
                reasons = []
                if default_profile:
                    reasons.append("• **Default Profile Settings:** The account uses default theme/profile settings, which is common in automated bot accounts.")
                else:
                    reasons.append("• **Customized Profile:** The account has customized its profile settings away from defaults.")

                if statuses_count > 10000 and account_age_days < 100:
                    reasons.append(f"• **High Activity Rate:** Extremely high tweet volume ({statuses_count:,} tweets) in a short account lifespan ({account_age_days} days).")
                
                if followers_count == 0 and friends_count > 500:
                    reasons.append("• **Imbalanced Follow Ratio:** Following a large number of accounts while having zero followers is an indicator of automated behavior.")
                elif followers_count > 10 and friends_count > 0:
                    ratio = followers_count / friends_count
                    if ratio > 0.2 and ratio < 5:
                        reasons.append(f"• **Healthy Follower/Following Ratio:** A balanced ratio ({ratio:.2f}) reflects organic social interaction.")

                if verified:
                    reasons.append("• **Account Verification:** The account is verified, heavily pointing toward a genuine human or official entity.")

                if len(description.strip()) < 5:
                    reasons.append("• **Empty or Minimal Bio:** The profile description is blank or extremely sparse.")
                else:
                    reasons.append(f"• **Detailed Bio Description:** Contains descriptive text ('{description[:40]}...').")

                if average_tweets_per_day > 50:
                    reasons.append(f"• **High Daily Frequency:** Averaging {average_tweets_per_day} tweets per day suggests automated scripting.")

                for r in reasons:
                    st.markdown(r)

            except Exception as e:
                st.error(f"Error during prediction: {e}")

with tab2:
    st.header("Dataset & Model Evaluation Inspector")
    st.write("Upload a CSV dataset to inspect its structure, or view the trained algorithm metrics and graphs below.")
    
    file = st.file_uploader("Upload a CSV file for inspection", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file, sep=None, engine="python", encoding="utf-8-sig")
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, sep=None, encoding="ISO-8859-1")
        a, b, c = st.columns(3)
        a.metric("Rows", f"{df.shape[0]:,}")
        b.metric("Columns", df.shape[1])
        c.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)
        st.subheader("Columns and Data Types")
        st.dataframe(pd.DataFrame({"column": df.columns, "dtype": [str(x) for x in df.dtypes], "missing": [int(x) for x in df.isna().sum()]}), use_container_width=True)
        label = next((c for c in df.columns if str(c).strip().lower() in {"account_type", "label", "bot", "class", "is_bot", "target"}), None)
        if label:
            st.subheader(f"Label Distribution: {label}")
            st.bar_chart(df[label].astype(str).value_counts())

    st.markdown("---")
    st.subheader("📈 Trained Algorithm Performance & Evaluation Graphs")
    
    if METRICS_PATH.exists():
        metrics_data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.write(f"**Selected Best Model:** `{metrics_data.get('selected_model_by_bot_f1')}`")
        
        # Display performance table for trained models
        models_perf = metrics_data.get("models", {})
        if models_perf:
            perf_rows = []
            for m_name, m_scores in models_perf.items():
                row = {"Model": m_name}
                row.update(m_scores)
                perf_rows.append(row)
            st.dataframe(pd.DataFrame(perf_rows), use_container_width=True)
    else:
        st.info("ℹ️ Run `python src/02_train_model.py` to generate model metrics.")

    if CONFUSION_MATRIX_PATH.exists():
        st.image(str(CONFUSION_MATRIX_PATH), caption="Model Confusion Matrix Evaluation Graph", use_container_width=True)
    else:
        st.info("ℹ️ Confusion matrix graph will appear here once the model training script has been executed.")