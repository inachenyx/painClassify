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
data = mat["allfeature"]

# Feature names: 39 per channel, 2 channels
features = ['max','min','mean','med','peak','arv','var','std','kurtosis','skewness','rms',
            'rs','rmsa','waveformF','peakF','impulseF','clearanceF','FC','MSF','RMSF','VF',
            'RVF','SKMean','SKStd','SKSkewness','SKKurtosis','psdE','svdpE','eE','ApEn',
            'SpEn','FuzzyEn','PeEn','enveEn','deltaDE','thetaDE','alphaDE','betaDE','gammaDE']
feature_names = [f"{name}_ch1" for name in features] + [f"{name}_ch2" for name in features]
# featuremap = {i: features[i] + "_ch1" if i <= 38 else features[i-39] + "_ch2" for i in range(78)}
# feature_names = [featuremap[i] for i in range(78)]

df = pd.DataFrame(data, columns=feature_names)

# labels: 20 subjects × 48 segments
labels = []
for subj in range(20):
    if subj < 7:
        labels += ["mild"] * 48
    elif subj < 14:
        labels += ["moderate"] * 48
    else:
        labels += ["severe"] * 48
df["label"] = labels

# ---------------------------
# Labels and groups
# ---------------------------
# y must be integers {0,1,2} for {mild, moderate, severe}
X_full = df.iloc[:, :-1].values
y_str = df["label"].values
# Encode labels to consecutive integers
le = LabelEncoder()
y = le.fit_transform(y_str)   # mild=0, moderate=1, severe=2
# Groups: default to 48 segments per subject if divisible
groups_full = np.repeat(np.arange(20), 48)   # subject IDs


# ---------------------------
# Task: moderate vs severe
# ---------------------------
mask = np.isin(y, [1, 2])
X_use = X_full[mask]
y_use = y[mask].copy()
y_use[y_use == 1] = 0
y_use[y_use == 2] = 1
groups_use = groups_full[mask]
y_enc = le.fit_transform(y_use)

# ---------------------------
# 2. Define models
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

# ---------------------------
# 3. Cross validation setup
# ---------------------------
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------------
# 4. Evaluate base models and build VotingClassifier
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
# 5.1 ET Top-20 feature importance with ExtraTrees
# ---------------------------
et_tmp = ExtraTreesClassifier(n_estimators=400, random_state=42, n_jobs=-1)
et_tmp.fit(X_use, y_enc)
importances = et_tmp.feature_importances_
idx = np.argsort(importances)[::-1]
sorted_f = [feature_names[i] for i in idx]

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
mean_scores = []
for k in range(5, 79, 5):
    X_topk = X_use[:, idx[:k]]
    scores = cross_val_score(et_tmp, X_topk, y_enc, cv=cv, groups=groups_use, scoring="accuracy")
    mean_scores.append(scores.mean())

plt.figure(figsize=(7,5))
plt.plot(range(5,79,5), mean_scores, marker="o")
plt.xlabel("Number of Top Features")
plt.ylabel("CV Accuracy (EF)")
plt.title("Feature Selection Trade-off")
plt.grid(True)
plt.show()

# ---------------------------
# 5.2 RF Feature importance + trade-off plot
# ---------------------------
rf = RandomForestClassifier(random_state=42, n_estimators=200)
rf.fit(X_use, y_enc)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = [feature_names[i] for i in indices]

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
mean_scores = []
for k in range(5, 79, 5):
    X_topk = X_use[:, indices[:k]]
    scores = cross_val_score(rf, X_topk, y_enc, cv=cv, groups=groups_use, scoring="accuracy")
    mean_scores.append(scores.mean())

plt.figure(figsize=(7,5))
plt.plot(range(5,79,5), mean_scores, marker="o")
plt.xlabel("Number of Top Features")
plt.ylabel("CV Accuracy (RF)")
plt.title("Feature Selection Trade-off")
plt.grid(True)
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
