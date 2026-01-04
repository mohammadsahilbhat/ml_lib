import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier

import warnings
warnings.filterwarnings("ignore")


# USER CONFIGURABLE PARAMETERS

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
TARGET = "target"    

N_SPLITS = 5
RANDOM_STATE = 42

K_BEST = 800          # mutual-info feature selection
PCA_COMPONENTS = 200  # PCA dimensionality, reduce noise

# LightGBM params
LGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 64,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# XGBoost params
XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# HGBT params
HGB_PARAMS = {
    "max_iter": 1000,
    "learning_rate": 0.03,
    "max_depth": 5,
    "random_state": RANDOM_STATE,
}



# LOAD DATA
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

X = train.drop(columns=[TARGET])
y = train[TARGET].values
X_test = test.copy()



# PREPROCESSING PIPELINE


# 1) Impute missing values
imputer = SimpleImputer(strategy="median")
X_imp = imputer.fit_transform(X)
X_test_imp = imputer.transform(X_test)

# 2) Variance threshold 
vt = VarianceThreshold(threshold=0.0)
X_vt = vt.fit_transform(X_imp)
X_test_vt = vt.transform(X_test_imp)

# rename columns (after vt names are lost)
X_vt_df = pd.DataFrame(X_vt, index=X.index)
X_test_vt_df = pd.DataFrame(X_test_vt, index=X_test.index)

# 3) Mutual Information feature selection 
k_best = min(K_BEST, X_vt_df.shape[1])
selector = SelectKBest(mutual_info_classif, k=k_best)
X_sel = selector.fit_transform(X_vt_df, y)
X_test_sel = selector.transform(X_test_vt_df)

# 4) PCA 
if PCA_COMPONENTS > 0:
    n_pc = min(PCA_COMPONENTS, X_sel.shape[1])
    pca = PCA(n_components=n_pc, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_sel)
    X_test_pca = pca.transform(X_test_sel)
else:
    X_pca = X_sel
    X_test_pca = X_test_sel

# 5) Standard Scaling
scaler = StandardScaler()
X_final = scaler.fit_transform(X_pca)
X_test_final = scaler.transform(X_test_pca)



# STACKING ENSEMBLE (OOF PREDICTIONS)

oof_preds = np.zeros((X_final.shape[0], 3))   # LGB, XGB, HGB
test_preds = np.zeros((X_test_final.shape[0], 3))

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for fold, (tr, val) in enumerate(skf.split(X_final, y)):
    print(f"=== Fold {fold+1}/{N_SPLITS} ===")
    X_tr, X_val = X_final[tr], X_final[val]
    y_tr, y_val = y[tr], y[val]

    # LIGHTGBM 
    lgb_clf = lgb.LGBMClassifier(**LGB_PARAMS)
    lgb_clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(0)
        ]
    )

    oof_preds[val, 0] = lgb_clf.predict_proba(X_val)[:,1]
    test_preds[:,0] += lgb_clf.predict_proba(X_test_final)[:,1] / N_SPLITS

    # XGBOOST 
    xgb_clf = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        
    )

    oof_preds[val, 1] = xgb_clf.predict_proba(X_val)[:,1]
    test_preds[:,1] += xgb_clf.predict_proba(X_test_final)[:,1] / N_SPLITS

    #  HGBT 
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS)
    hgb.fit(X_tr, y_tr)

    oof_preds[val, 2] = hgb.predict_proba(X_val)[:,1]
    test_preds[:,2] += hgb.predict_proba(X_test_final)[:,1] / N_SPLITS

    # Quick fold check
    fold_pred = (oof_preds[val].mean(axis=1) > 0.5).astype(int)
    fold_ba = balanced_accuracy_score(y_val, fold_pred)
    print(f"Fold Balanced Accuracy: {fold_ba:.4f}")



# META MODEL (STACKING)

meta = LogisticRegression(max_iter=2000, class_weight="balanced")
meta.fit(oof_preds, y)

oof_meta_proba = meta.predict_proba(oof_preds)[:,1]
test_meta_proba = meta.predict_proba(test_preds)[:,1]



# OPTIMIZE THRESHOLD FOR BALANCED ACCURACY
def find_best_threshold(y_true, probs):
    best_thr, best_ba = 0.5, 0
    for thr in np.linspace(0.01, 0.99, 200):
        preds = (probs > thr).astype(int)
        ba = balanced_accuracy_score(y_true, preds)
        if ba > best_ba:
            best_ba = ba
            best_thr = thr
    return best_thr, best_ba

best_thr, best_oof_ba = find_best_threshold(y, oof_meta_proba)

print(f"\nBest threshold = {best_thr:.4f}")
print(f"OOF Balanced Accuracy = {best_oof_ba:.6f}")


# FINAL TEST PREDICTION

final_test_pred = (test_meta_proba > best_thr).astype(int)

submission = pd.DataFrame ({"id": test['id'],
    "target": final_test_pred
})

submission.to_csv("submission.csv", index=False)

print("\nCompleted! Submission saved to 'submission.csv'.")
