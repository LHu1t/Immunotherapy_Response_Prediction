import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# 1. LOAD DATA
# ----------------------------
df = pd.read_csv("LUAD_LUSC_Data/combined_clinical_expression_ALL2.csv")

# === Step 2: Detect the cancer subtype ===
diagnosis_col = None
for col in df.columns:
    if "primary_diagnosis" in col.lower():
        diagnosis_col = col
        break

if diagnosis_col is None:
    raise ValueError("Could not find a column containing 'primary_diagnosis'")

def classify_subtype(x):
    if isinstance(x, str):
        x = x.lower()
        if "adeno" in x or "luad" in x:
            return "Adenocarcinoma"
        elif "squamous" in x or "lusc" in x:
            return "Squamous"
    return None

df["Cancer_Subtype"] = df[diagnosis_col].apply(classify_subtype)
df = df[df["Cancer_Subtype"].notnull()]  # keep only valid subtypes

print(f"\nDetected {df['Cancer_Subtype'].value_counts().to_dict()}")

# ----------------------------
# 3. SELECT EXPRESSION FEATURES
# ----------------------------
expr_cols = [c for c in df.columns if c.endswith("_fpkm_uq")]
if not expr_cols:
    raise ValueError("No gene expression columns ending with '_fpkm_uq' found!")

# Drop rows missing expression values
df_filtered = df.dropna(subset=expr_cols)

# Show number of patients
n_total = len(df)
n_filtered = len(df_filtered)
print(f"\nTotal patients: {n_total}")
print(f"Patients used (non-missing expression): {n_filtered}")
print(f"→ {n_filtered/n_total*100:.1f}% retained for analysis")

# ----------------------------
# 4. PREPROCESS FEATURES
# ----------------------------
X = np.log2(df_filtered[expr_cols] + 1)  # log2 transform
y = df_filtered["Cancer_Subtype"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 5. TRAIN/VALIDATION SPLIT
# ----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# ----------------------------
# 6. RANDOM FOREST CLASSIFIER
# ----------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    max_depth=None
)

rf.fit(X_train, y_train)

# ----------------------------
# 7. EVALUATION
# ----------------------------
y_pred = rf.predict(X_val)
y_proba = rf.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_proba)

print(f"\nAccuracy: {acc:.3f}")
print(f"ROC-AUC: {roc_auc:.3f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Subtype Prediction)")
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_val, y_proba)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Cancer Subtype Prediction")
plt.legend()
plt.show()

# ----------------------------
# 8. FEATURE IMPORTANCE
# ----------------------------
importances = pd.Series(rf.feature_importances_, index=expr_cols)
top_genes = importances.sort_values(ascending=False).head(10)
print("\nTop 10 most predictive genes:")
print(top_genes)

plt.figure(figsize=(7,4))
sns.barplot(x=top_genes.values, y=top_genes.index)
plt.title("Top 10 Genes Driving Subtype Prediction")
plt.xlabel("Feature Importance (Random Forest)")
plt.show()

# ----------------------------
# 9. SAVE PREDICTIONS
# ----------------------------
df_results = df_filtered.copy()
probs = rf.predict_proba(X_scaled)
df_results["Predicted_Subtype"] = le.inverse_transform(rf.predict(X_scaled))
df_results[f"Prob_{le.classes_[0]}"] = probs[:, 0]
df_results[f"Prob_{le.classes_[1]}"] = probs[:, 1]

df_results.to_csv("subtype_predictions.csv", index=False)
print("\n Saved predictions to 'subtype_predictions.csv'")
