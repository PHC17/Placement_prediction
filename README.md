# 🎓 Student Placement Prediction System

A 3-page Streamlit app (Home · Live Predictor · Insights) backed by a scikit-learn model trained on 100,000 student records.

## Run it
```bash
pip install -r requirements.txt
python train.py path/to/student_placement_prediction_dataset_2026.csv   # optional re-train
streamlit run Home.py
```
Pre-trained artefacts are already included, so `streamlit run Home.py` works immediately.

## Structure
```
Home.py                 Landing page (hero, stats, categories, how-it-works, tech stack)
pages/1_Predictor.py    Live form, sample profiles, prediction, risk band, recommendations
pages/2_Insights.py     EDA charts, model comparison, confusion matrix, feature importance, limits
utils/theme.py          CSS, light/dark toggle, Plotly styling, HTML card helpers
utils/predictor.py      Loading, validation, prediction, risk bands, recommendations (no UI code)
train.py                Full training pipeline → models/*.joblib, metrics, EDA aggregates
models/                 preprocessor.joblib, model.joblib, meta.json, metrics CSVs
data/eda_sample.csv     6,000-row sample used by Insights charts
```

## Design notes
* `student_id` and `salary_package_lpa` (post-placement leakage) are never used. `age` and `gender` are also excluded.
* Same `ColumnTransformer` (StandardScaler + OneHotEncoder) is saved and re-used at prediction time.
* The best model is chosen by ROC-AUC; the label uses a balanced decision threshold (the training placement rate).
* Risk bands (Low / Moderate / High) are tertiles of predicted probability on the test set.
* Dark mode is CSS-based (Streamlit has no runtime theme switch), toggled from the sidebar.
