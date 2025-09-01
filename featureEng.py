# ======= feature_tradeoff_analysis.py =======
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, balanced_accuracy_score

# ----------------------------
# 0) Config toggles
# ----------------------------
RANDOM_STATE = 42

# classifier options: "extratrees", "logreg", "lda"
CLASSIFIER = ""

# ranking options for cumulative-importance figure:
# "extratrees_impurity" or "cv_permutation"
RANKING_FOR_CUMULATIVE = "extratrees_impurity"

# ranking used inside the pipeline selector for perf-vs-k:
# "extratrees_impurity" (fast) or "rf_impurity" (also ok)
SELECTOR_RANKING = "extratrees_impurity"

# scoring: "accuracy" or "balanced_accuracy"
SCORING = "balanced_accuracy"

# smoothing window for the overlay trend on perf-vs-k (set to 0 to disable)
SMOOTH_WINDOW = 3  # moving average window (odd integer recommended)
# ----------------------------


# ----------------------------
# A) Load data + build labels
# ----------------------------
mat = sio.loadmat("allfeature.mat")
X_all = mat["allfeature"]  # shape (960, 78)

# Recreate feature names as in your wilcoxon.py
features = ['max','min','mean','med','peak','arv','var','std','kurtosis','skewness','rms',
            'rs','rmsa','waveformF','peakF','impulseF','clearanceF','FC','MSF','RMSF','VF',
            'RVF','SKMean','SKStd','SKSkewness','SKKurtosis','psdE','svdpE','eE','ApEn',
            'SpEn','FuzzyEn','PeEn','enveEn','detaDE','thetaDE','alphaDE','betaDE','gammaDE']
featuremap = {i: features[i] + "_ch1" if i <= 38 else features[i-39] + "_ch2" for i in range(78)}
feature_names = [featuremap[i] for i in range(78)]
df = pd.DataFrame(X_all, columns=feature_names)

# Pain labels: 20 subjects × 48 segments each
labels = []
for subj in range(20):
    if subj < 7:
        labels += ["mild"] * 48
    elif subj < 14:
        labels += ["moderate"] * 48
    else:
        labels += ["severe"] * 48
df["label"] = labels

# Encode y
le = LabelEncoder()
y = le.fit_transform(df["label"].values)  # mild=0, moderate=1, severe=2
X = df.iloc[:, :-1].values

# Subject groups for GroupKFold
groups = np.repeat(np.arange(20), 48)  # 0..19 repeated 48 times


# ----------------------------
# B) Helpers
# ----------------------------
def get_classifier(kind=CLASSIFIER, random_state=RANDOM_STATE):
    if kind == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=400,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=2,
            class_weight="balanced"
        )
    elif kind == "logreg":
        # logreg prefers scaled features
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=3000,
                multi_class="auto",
                class_weight="balanced",
                random_state=random_state
            ))
        ])
    elif kind == "lda":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LDA())
        ])
    else:
        raise ValueError("Unknown classifier")

def get_importance_ranking(X, y, method="extratrees_impurity", groups=None, random_state=RANDOM_STATE):
    """
    Returns (sorted_indices, importances) where importances are aligned with feature_names.
    """
    if method == "extratrees_impurity":
        model = ExtraTreesClassifier(
            n_estimators=600, random_state=random_state, n_jobs=-1, min_samples_leaf=2,
            class_weight="balanced"
        )
        model.fit(X, y)
        imps = model.feature_importances_
        order = np.argsort(imps)[::-1]
        return order, imps

    elif method == "rf_impurity":
        model = RandomForestClassifier(
            n_estimators=600, random_state=random_state, n_jobs=-1, min_samples_leaf=2,
            class_weight="balanced"
        )
        model.fit(X, y)
        imps = model.feature_importances_
        order = np.argsort(imps)[::-1]
        return order, imps

    elif method == "cv_permutation":
        # Aggregate permutation importance across StratifiedGroupKFold
        # This is slower but more stable.
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
        imps_accum = np.zeros(X.shape[1], dtype=float)
        counts = np.zeros(X.shape[1], dtype=int)
        base = get_classifier("extratrees", random_state=random_state)

        for train_idx, test_idx in cv.split(X, y, groups):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            model = clone(base)
            model.fit(X_tr, y_tr)
            # Permutation importance computed on the validation fold
            pi = permutation_importance(model, X_te, y_te, n_repeats=10, random_state=random_state, n_jobs=-1)
            imps_accum += pi.importances_mean
            counts += 1
        imps = imps_accum / np.maximum(counts, 1)
        order = np.argsort(imps)[::-1]
        return order, imps

    else:
        raise ValueError("Unknown importance method")

class SelectTopKFromModel(BaseEstimator, TransformerMixin):
    """
    Fits a tree model on the training data and selects top-k features by impurity importances.
    This happens within each CV split (to avoid leakage).
    """
    def __init__(self, base_estimator=None, k=20, random_state=RANDOM_STATE):
        self.base_estimator = base_estimator if base_estimator is not None else ExtraTreesClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1, min_samples_leaf=2,
            class_weight="balanced"
        )
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        self.est_ = clone(self.base_estimator)
        self.est_.fit(X, y)
        imps = self.est_.feature_importances_
        self.order_ = np.argsort(imps)[::-1]
        self.keep_idx_ = self.order_[:self.k]
        return self

    def transform(self, X):
        return X[:, self.keep_idx_]


def moving_average(y, window=3):
    if window is None or window < 2:
        return y
    kernel = np.ones(window) / window
    z = np.convolve(y, kernel, mode="valid")
    # pad ends to original length
    pad_left = window//2
    pad_right = len(y) - len(z) - pad_left
    return np.pad(z, (pad_left, pad_right), mode="edge")


# ----------------------------
# C) Figure 1 — Cumulative importance
# ----------------------------
def plot_cumulative_importance(X, y, method=RANKING_FOR_CUMULATIVE, groups=groups,
                               fname="cumulative_importance.png"):
    order, imps = get_importance_ranking(X, y, method=method, groups=groups, random_state=RANDOM_STATE)
    sorted_imps = imps[order]
    cum = np.cumsum(sorted_imps)

    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(sorted_imps)+1), cum, marker='o')
    plt.axhline(0.7, linestyle='--')
    # Annotate the smallest k reaching 0.7
    k70 = int(np.argmax(cum >= 0.7) + 1)
    plt.axvline(k70, linestyle='--')
    plt.title(f"Cumulative Feature Importance ({method})")
    plt.xlabel("Number of top features")
    plt.ylabel("Cumulative importance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()

    # Export CSV
    out = pd.DataFrame({
        "rank": np.arange(1, len(order)+1),
        "feature": np.array(feature_names)[order],
        "importance": sorted_imps,
        "cumulative_importance": cum
    })
    out.to_csv("cumulative_importance_" + method + ".csv", index=False)
    print(f"[Saved] {fname} and cumulative_importance_{method}.csv (k@70% = {k70})")


# ----------------------------
# D) Figure 2 — Perf vs feature count
# ----------------------------
def perf_vs_k_plot(X, y, groups, selector_ranking=SELECTOR_RANKING,
                   classifier=CLASSIFIER, fname="performance_vs_features.png",
                   scoring=SCORING, smooth_window=SMOOTH_WINDOW):
    # base selector model for impurity importances
    if selector_ranking == "extratrees_impurity":
        selector_base = ExtraTreesClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=2,
            class_weight="balanced"
        )
    elif selector_ranking == "rf_impurity":
        selector_base = RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=2,
            class_weight="balanced"
        )
    else:
        raise ValueError("selector_ranking should be 'extratrees_impurity' or 'rf_impurity' for pipeline selector.")

    # classifier
    clf = get_classifier(classifier, random_state=RANDOM_STATE)

    # scoring
    scorer = (make_scorer(balanced_accuracy_score) if scoring == "balanced_accuracy" else scoring)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    ks = list(range(5, 79, 5))
    means, stds = [], []

    for k in ks:
        pipe = Pipeline([
            ("selectk", SelectTopKFromModel(base_estimator=selector_base, k=k, random_state=RANDOM_STATE)),
            ("clf", clf)
        ])
        scores = []
        for tr, te in cv.split(X, y, groups):
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
            s = cross_val_score(pipe, Xtr, ytr, cv=3, scoring=scorer, n_jobs=-1).mean()
            # evaluate on held-out fold to be conservative
            pipe.fit(Xtr, ytr)
            if scoring == "balanced_accuracy":
                ypred = pipe.predict(Xte)
                s_holdout = balanced_accuracy_score(yte, ypred)
            else:
                s_holdout = pipe.score(Xte, yte)
            # combine inner-CV estimate and holdout estimate (optional): here we just use holdout to reduce optimism
            scores.append(s_holdout)
        means.append(np.mean(scores))
        stds.append(np.std(scores))

    means = np.array(means)
    stds = np.array(stds)

    plt.figure(figsize=(8,6))
    plt.errorbar(ks, means, yerr=stds, fmt='-o', capsize=3)
    plt.xlabel("Number of top features (k)")
    plt.ylabel(f"{'Balanced ' if scoring=='balanced_accuracy' else ''}Accuracy (CV mean ± SD)")
    plt.title("Performance vs Feature Count (Group-aware CV)")
    plt.grid(True)

    # optional smoothing overlay (clearly labeled)
    if smooth_window and smooth_window >= 2:
        sm = moving_average(means, window=smooth_window)
        plt.plot(ks, sm, linestyle='--', label=f"Moving average (w={smooth_window})")
        plt.legend()

    # Heuristic: mark smallest k within 1 SD of best mean
    best = means.max()
    within = np.where(means >= (best - stds[np.argmax(means)]))[0]
    if len(within) > 0:
        k_star = ks[int(within[0])]
        plt.axvline(k_star, linestyle='--')
        plt.text(k_star, best, f"k*≈{k_star}", ha="left", va="bottom")

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()

    out = pd.DataFrame({"k": ks, "mean": means, "std": stds})
    out.to_csv("performance_vs_features.csv", index=False)
    print(f"[Saved] {fname} and performance_vs_features.csv")


# ----------------------------
# Run all
# ----------------------------
if __name__ == "__main__":
    # Figure 1
    plot_cumulative_importance(X, y, method=RANKING_FOR_CUMULATIVE, groups=groups,
                               fname="cumulative_importance.png")

    # Figure 2
    perf_vs_k_plot(X, y, groups,
                   selector_ranking=SELECTOR_RANKING,
                   classifier=CLASSIFIER,
                   fname="performance_vs_features.png",
                   scoring=SCORING,
                   smooth_window=SMOOTH_WINDOW)

    print("Done.")

