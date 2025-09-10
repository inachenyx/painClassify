import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, cross_val_predict, RandomizedSearchCV, learning_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import randint, uniform, loguniform
np.random.seed(42)

# ---------------------------
# 1. Load and prepare data
# ---------------------------
mat = sio.loadmat("allfeature.mat")
if "allfeature" in mat:
    X_full = mat["allfeature"]
elif "X" in mat:
    X_full = mat["X"]
else:
    raise RuntimeError("Expected 'allfeature' in allfeature.mat")

N, D = X_full.shape
if D != 78:
    print(f"Warning: expected 78 features, got {D}")

# Feature names: 39 per channel, 2 channels
features = ['max','min','mean','med','peak','arv','var','std','kurtosis','skewness','rms',
            'rs','rmsa','waveformF','peakF','impulseF','clearanceF','FC','MSF','RMSF','VF',
            'RVF','SKMean','SKStd','SKSkewness','SKKurtosis','psdE','svdpE','eE','ApEn',
            'SpEn','FuzzyEn','PeEn','enveEn','deltaDE','thetaDE','alphaDE','betaDE','gammaDE']
feature_names = [f"{name}_ch1" for name in features] + [f"{name}_ch2" for name in features]

# ---------------------------
# Labels and groups
# ---------------------------
# y must be integers {0,1,2} for {mild, moderate, severe}
# Try to locate labels nearby
y = None
labels_paths = ["labels.npy", "labels.mat", "y.mat", "labels.csv"]
for lp in labels_paths:
    p = Path(lp)
    if p.exists():
        if lp.endswith(".npy"):
            y = np.load(p)
        elif lp.endswith(".csv"):
            df = pd.read_csv(p)
            if "label" not in df.columns:
                raise RuntimeError("labels.csv must have a 'label' column")
            y = df["label"].to_numpy()
        else:
            md = sio.loadmat(p)
            for key in ["labels","y","label","Y"]:
                if key in md:
                    y = md[key].ravel()
                    break
        if y is not None:
            break

if y is None:
    raise RuntimeError("Could not find labels. Place labels.npy or labels.mat or labels.csv next to allfeature.mat")

y = np.asarray(y).ravel()
if y.shape[0] != X_full.shape[0]:
    raise ValueError(f"Labels length {y.shape[0]} does not match X rows {X_full.shape[0]}")

# Groups: default to 48 segments per subject if divisible
if N % 48 == 0:
    n_subjects = N // 48
    groups_full = np.repeat(np.arange(n_subjects), 48)
else:
    groups_full = np.arange(N)

# Encode labels to consecutive integers
le = LabelEncoder()
# ---------------------------
# Task: original three-class with SVM added
# ---------------------------
mask = np.isin(y, [0, 1, 2])
X_use = X_full[mask]
y_use = y[mask].copy()
groups_use = groups_full[mask]
y_enc = le.fit_transform(y_use)

# ---------------------------
# 2. Define models, add SVM with tuning
# ---------------------------
models = {
    "LDA": LinearDiscriminantAnalysis(),
    "DT": DecisionTreeClassifier(random_state=42),
    "RF": RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
    "ET": ExtraTreesClassifier(n_estimators=400, random_state=42, n_jobs=-1),
    "GBDT": GradientBoostingClassifier(random_state=42),
    "GNB": GaussianNB(),
    "KNN": KNeighborsClassifier()
}

# Add tuned SVM via pipeline
svm_pipe = Pipeline([("scaler", StandardScaler(with_mean=True)), ("svc", SVC(probability=True, random_state=42))])
svm_dist = {
    "svc__kernel": ["rbf", "linear"],
    "svc__C": loguniform(1e-2, 1e3),
    "svc__gamma": loguniform(1e-4, 1e0)
}

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
svm_search = RandomizedSearchCV(
    svm_pipe, param_distributions=svm_dist, n_iter=25, cv=cv, random_state=42,
    scoring="balanced_accuracy", n_jobs=-1, refit=True, verbose=0
)
svm_search.fit(X_use, y_enc, groups=groups_use)
print("SVM best params:", svm_search.best_params_)
models["SVM"] = svm_search.best_estimator_

# ---------------------------
# 3. Evaluate base models and build VotingClassifier
# ---------------------------
scores_summary = {}
for name, est in models.items():
    sc = cross_val_score(est, X_use, y_enc, cv=cv, groups=groups_use, scoring="balanced_accuracy", n_jobs=-1)
    scores_summary[name] = (sc.mean(), sc.std())
    print(f"{name:>5}: mean={sc.mean():.3f} ± {sc.std():.3f}")

estimators = [(k, v) for k, v in models.items()]
supports_proba = all(hasattr(est, "predict_proba") for _, est in estimators)
voting = "soft" if supports_proba else "hard"
final_vc = VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)

# ---------------------------
# 5. Top-20 feature importance with ExtraTrees
# ---------------------------
et_tmp = ExtraTreesClassifier(n_estimators=400, random_state=42, n_jobs=-1)
et_tmp.fit(X_use, y_enc)
importances = et_tmp.feature_importances_
idx_top20 = np.argsort(importances)[-20:][::-1]
plt.figure(figsize=(10,5))
plt.bar(range(20), importances[idx_top20])
plt.xticks(range(20), [feature_names[i] for i in idx_top20], rotation=60, ha='right')
plt.ylabel("Importance")
plt.title("Top-20 Feature Importances (ExtraTrees)")
plt.tight_layout()
plt.show()

# ---------------------------
# 6. Learning curve for VotingClassifier
# ---------------------------
train_sizes, train_scores, valid_scores = learning_curve(
    final_vc, X_use, y_enc, cv=cv, groups=groups_use, scoring="balanced_accuracy", n_jobs=-1,
    train_sizes=np.linspace(0.2, 1.0, 5), shuffle=True, random_state=42
)
train_mean = train_scores.mean(axis=1)
valid_mean = valid_scores.mean(axis=1)

plt.figure(figsize=(7,5))
plt.plot(train_sizes, train_mean, marker='o', label="Training score")
plt.plot(train_sizes, valid_mean, marker='s', label="CV score")
plt.xlabel("Training set size")
plt.ylabel("Balanced accuracy")
plt.title("Learning Curve (VotingClassifier)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# 7. Confusion Matrix for VotingClassifier (CV predict)
# ---------------------------
y_cv_pred = cross_val_predict(final_vc, X_use, y_enc, cv=cv, groups=groups_use, n_jobs=-1)
cm = confusion_matrix(y_enc, y_cv_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (CV prediction)")
plt.tight_layout()
plt.show()
