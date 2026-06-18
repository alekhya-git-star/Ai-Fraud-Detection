import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

print("="*60)
print("FRAUD DETECTION - FULL TRAINING PIPELINE")
print("="*60)

# ── 1. Load data ──────────────────────────────────────────────
print("\n[1/7] Loading dataset...")
df = pd.read_csv('dataset/credit_card_transactions.csv')
print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")

# ── 2. Feature Engineering ────────────────────────────────────
print("\n[2/7] Feature engineering...")

# Parse datetime
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour']        = df['trans_date_trans_time'].dt.hour
df['day']         = df['trans_date_trans_time'].dt.day
df['month']       = df['trans_date_trans_time'].dt.month
df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek

# Age from DOB
df['dob'] = pd.to_datetime(df['dob'])
df['age'] = ((df['trans_date_trans_time'] - df['dob']).dt.days / 365.25).astype(int)

# Distance between cardholder and merchant
df['distance'] = np.sqrt(
    (df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2
)

# Encode categoricals
from sklearn.preprocessing import LabelEncoder

le_merchant  = LabelEncoder()
le_category  = LabelEncoder()
le_gender    = LabelEncoder()
le_city      = LabelEncoder()
le_state     = LabelEncoder()
le_job       = LabelEncoder()

df['merchant_enc'] = le_merchant.fit_transform(df['merchant'].astype(str))
df['category_enc'] = le_category.fit_transform(df['category'].astype(str))
df['gender_enc']   = le_gender.fit_transform(df['gender'].astype(str))
df['city_enc']     = le_city.fit_transform(df['city'].astype(str))
df['state_enc']    = le_state.fit_transform(df['state'].astype(str))
df['job_enc']      = le_job.fit_transform(df['job'].astype(str))

# Save categories for app
categories = sorted(df['category'].unique().tolist())
joblib.dump(categories, 'categories.pkl')
print(f"  Categories: {categories}")

# Feature set
FEATURES = [
    'merchant_enc', 'category_enc', 'amt', 'gender_enc',
    'city_enc', 'state_enc', 'zip', 'lat', 'long',
    'city_pop', 'job_enc', 'merch_lat', 'merch_long',
    'hour', 'day', 'month', 'day_of_week', 'age', 'distance'
]

joblib.dump(FEATURES, 'feature_columns.pkl')
print(f"  Features: {len(FEATURES)}")

X = df[FEATURES]
y = df['is_fraud']
print(f"  Fraud rate: {y.mean()*100:.2f}%  ({y.sum():,} fraud / {len(y):,} total)")

# ── 3. Train/Test Split ───────────────────────────────────────
print("\n[3/7] Splitting data (80/20)...")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 4. Handle class imbalance with undersampling ──────────────
print("\n[4/7] Balancing classes (RandomUnderSampler)...")
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)
print(f"  Resampled train: {len(X_res):,}  |  Fraud: {y_res.sum():,}  |  Legit: {(y_res==0).sum():,}")

# ── 5. Train Random Forest ────────────────────────────────────
print("\n[5/7] Training Random Forest...")
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_res, y_res)
print("  Training complete.")

# ── 6. Evaluate ───────────────────────────────────────────────
print("\n[6/7] Evaluating on test set...")
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test, y_pred)
auc     = roc_auc_score(y_test, y_proba)
f1      = f1_score(y_test, y_pred)
prec    = precision_score(y_test, y_pred)
rec     = recall_score(y_test, y_pred)
cm      = confusion_matrix(y_test, y_pred)

print(f"\n  ┌─────────────────────────────┐")
print(f"  │  AUC-ROC   : {auc*100:.2f}%          │")
print(f"  │  Accuracy  : {acc*100:.2f}%          │")
print(f"  │  F1 Score  : {f1*100:.2f}%          │")
print(f"  │  Precision : {prec*100:.2f}%          │")
print(f"  │  Recall    : {rec*100:.2f}%          │")
print(f"  └─────────────────────────────┘")
print(f"\n  Confusion Matrix:\n{cm}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Legit','Fraud'])}")

# Save metrics
metrics = {
    'auc': auc, 'accuracy': acc, 'f1': f1,
    'precision': prec, 'recall': rec,
    'confusion_matrix': cm.tolist(),
    'train_size': len(X_res), 'test_size': len(X_test),
    'total_transactions': len(df),
    'fraud_count': int(y.sum()),
    'fraud_rate': float(y.mean())
}
joblib.dump(metrics, 'model_metrics.pkl')

# ── 7. Save model & encoders ──────────────────────────────────
print("\n[7/7] Saving model and encoders...")
joblib.dump(model, 'fraud_model.pkl')
encoders = {
    'merchant': le_merchant, 'category': le_category,
    'gender': le_gender, 'city': le_city,
    'state': le_state, 'job': le_job
}
joblib.dump(encoders, 'label_encoders.pkl')
print("  Saved: fraud_model.pkl, label_encoders.pkl, categories.pkl, feature_columns.pkl, model_metrics.pkl")

# ── SHAP ──────────────────────────────────────────────────────
print("\n[BONUS] Generating SHAP feature importance...")
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(model)
X_shap = X_test.sample(500, random_state=42)
shap_values = explainer.shap_values(X_shap)

sv = shap_values[1] if isinstance(shap_values, list) else shap_values[:, :, 1]

plt.figure(figsize=(10, 6))
plt.style.use('dark_background')
shap.summary_plot(sv, X_shap, feature_names=FEATURES,
                  plot_type='bar', show=False, color='#38bdf8')
plt.title('SHAP Feature Importance — Fraud Detection', color='white', fontsize=13)
plt.tight_layout()
plt.savefig('shap_plot.png', dpi=150, bbox_inches='tight',
            facecolor='#0b0f1a', edgecolor='none')
plt.close()
print("  Saved: shap_plot.png")

# ── Power BI export ───────────────────────────────────────────
print("\n[BONUS] Generating Power BI export...")
df['fraud_prob'] = model.predict_proba(df[FEATURES])[:, 1]
df['predicted_fraud'] = model.predict(df[FEATURES])
df['fraud_label'] = df['is_fraud'].map({0: 'Legitimate', 1: 'Fraudulent'})
df['predicted_label'] = df['predicted_fraud'].map({0: 'Legitimate', 1: 'Fraudulent'})
df['risk_level'] = pd.cut(df['fraud_prob'],
                           bins=[0, 0.25, 0.6, 1.0],
                           labels=['Low', 'Medium', 'High'])

powerbi_cols = [
    'trans_date_trans_time', 'merchant', 'category', 'amt',
    'gender', 'city', 'state', 'zip', 'city_pop', 'job',
    'hour', 'day', 'month', 'day_of_week', 'age', 'distance',
    'is_fraud', 'fraud_label', 'fraud_prob', 'predicted_fraud',
    'predicted_label', 'risk_level', 'lat', 'long', 'merch_lat', 'merch_long'
]
df[powerbi_cols].to_csv('powerbi_data.csv', index=False)
print(f"  Saved: powerbi_data.csv ({len(df):,} rows)")

print("\n" + "="*60)
print("PIPELINE COMPLETE ✅")
print("="*60)
