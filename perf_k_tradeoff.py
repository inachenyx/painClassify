import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
from scipy.io import loadmat

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone

# ----------------------------
# Load your data
# ----------------------------
mat = sio.loadmat("allfeature.mat")
X_all = mat["allfeature"]

# Feature names (same mapping as before)
features = ['max', 'min', 'mean', 'med', 'peak', 'arv', 'var', 'std', 'kurtosis', 'skewness', 'rms',
            'rs', 'rmsa', 'waveformF', 'peakF', 'impulseF', 'clearanceF', 'FC', 'MSF', 'RMSF', 'VF',
            'RVF', 'SKMean', 'SKStd', 'SKSkewness', 'SKKurtosis', 'psdE', 'svdpE', 'eE', 'ApEn',
            'SpEn', 'FuzzyEn', 'PeEn', 'enveEn', 'detaDE', 'thetaDE', 'alphaDE', 'betaDE', 'gammaDE']
featuremap = {i: features[i] + "_ch1" if i <= 38 else features[i - 39] + "_ch2" for i in range(78)}
feature_names = [featuremap[i] for i in range(78)]

# Labels
labels = []
for subj in range(20):
    if subj < 7:
        labels += ["mild"] * 48
    elif subj < 14:
        labels += ["moderate"] * 48
    else:
        labels += ["severe"] * 48

le = LabelEncoder()
y = le.fit_transform(labels)
X = X_all
groups = np.repeat(np.arange(20), 48)


# ----------------------------
# Choose classifier
# ----------------------------
def get_clf(name="extratrees"):
    if name == "extratrees":
        return ExtraTreesClassifier(n_estimators=600, random_state=42, n_jobs=-1, class_weight="balanced")
    elif name == "rf":
        return RandomForestClassifier(n_estimators=600, random_state=42, n_jobs=-1, class_weight="balanced")
    elif name == "logreg":
        return LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)
    elif name == "lda":
        return LDA()
    else:
        raise ValueError


clf_name = "extratrees"  # try "logreg" or "lda" too
clf = get_clf(clf_name)

# ----------------------------
# Perf vs k
# ----------------------------
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
ks = list(range(5, 79, 5))
means, stds = [], []

for k in ks:
    scores = []
    for tr, te in cv.split(X, y, groups):
        # Fit selector on training
        selector = ExtraTreesClassifier(n_estimators=600, random_state=42, n_jobs=-1, class_weight="balanced")
        selector.fit(X[tr], y[tr])
        topk_idx = np.argsort(selector.feature_importances_)[::-1][:k]

        # Train classifier on selected features
        model = clone(clf)
        model.fit(X[tr][:, topk_idx], y[tr])

        # Evaluate on held-out
        y_pred = model.predict(X[te][:, topk_idx])
        score = balanced_accuracy_score(y[te], y_pred)
        scores.append(score)
    means.append(np.mean(scores))
    stds.append(np.std(scores))

plt.errorbar(ks, means, yerr=stds, fmt="-o", capsize=3)
plt.xlabel("Number of top features")
plt.ylabel("Balanced accuracy (mean ± SD)")
plt.title(f"Perf vs k ({clf_name})")
plt.grid(True)
plt.show()

pd.DataFrame({"k": ks, "mean": means, "std": stds}).to_csv("perf_vs_k_simple.csv", index=False)
