import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# -----------------------------
# Load data
# -----------------------------
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

X = train_df.drop("target", axis=1)
y = train_df["target"]

# -----------------------------
# Train / Validation split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Pipeline: Scaling + Logistic Regression
# -----------------------------
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        penalty="l2",          # regularization
        C=1.0,                 # inverse regularization strength
        solver="saga",         # best for many features
        max_iter=5000,
        n_jobs=-1
    ))
])

# -----------------------------
# Train
# -----------------------------
pipe.fit(X_train, y_train)

# -----------------------------
# Validation performance
# -----------------------------
y_val_pred = pipe.predict(X_val)
y_val_prob = pipe.predict_proba(X_val)[:, 1]

print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("ROC-AUC :", roc_auc_score(y_val, y_val_prob))
print(classification_report(y_val, y_val_pred))

# -----------------------------
# Predict on test set
# -----------------------------
test_preds = pipe.predict(test_df)
two_cols = test_preds[["id", "target"]]

two_cols.to_csv('submission.csv',index=False)
