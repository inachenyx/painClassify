# -*- coding: utf-8 -*-
# End-to-end pipeline: feature ranking -> model tuning (RandomizedSearchCV) -> voting -> plots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    StratifiedGroupKFold, cross_val_score, cross_val_predict,
    RandomizedSearchCV, learning_curve
)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay, make_scorer
)

from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from scipy.stats import randint, uniform, reciprocal

# ---------------------------
# 1) Load data & labels
# ---------------------------
mat = sio.loadmat("allfeature.mat")  # change path if needed
X_all = mat["allfeature"]  # shape (960, 78)

# Build feature names
base_feats = ['max','min','mean','med','peak','arv','var','std','kurtosis','skewness','rms',
              'rs','rmsa','waveformF','peakF','impulseF','clearanceF','FC','MSF','RMSF','VF',
              'RVF','SKMean','SKStd','SKSkewness','SKKurtosis','psdE','svdpE','eE','ApEn',
              'SpEn','FuzzyEn','PeEn','enveEn','detaDE','thetaDE','alphaDE','betaDE','gammaDE']
feature_names = [ (base_feats[i] + "_ch1") if i <= 38 else (base_feats[i-39] + "_ch2")
                  for i in range(78) ]
df = pd.DataFrame(X_all, columns=feature_names)

# Subject-grouped labels (20 subjects * 48 segments = 960 rows)
labels = []
for subj in range(20):
    labels += (["mild"]*48 if subj < 7 else ["moderate"]*48 if subj < 14 else ["severe"]*48)
df["label"] = labels

X = df.iloc[:, :-1].values
y_str = df["label"].values
le = LabelEncoder()
y = le.fit_transform(y_str)  # mild=0, moderate=1, severe=2
groups = np.repeat(np.arange(20), 48)  # group by subject

# ---------------------------
# 2) Feature ranking + trade-off curve (RF, balanced accuracy)
# ---------------------------
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
bal_scorer = make_scorer(balanced_accuracy_score)

rf_ranker = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)
rf_ranker.fit(X, y)
importances = rf_ranker.feature_importances_
rank_idx = np.argsort(importances)[::-1]

# Trade-off: CV balanced accuracy vs top-k features (RF as proxy)
k_vals = list(range(5, 79, 5))
tradeoff_means = []
for k in k_vals:
    Xk = X[:, rank_idx[:k]]
    scores = cross_val_score(rf_ranker, Xk, y, cv=cv, groups=groups, scoring=bal_scorer, n_jobs=-1)
    tradeoff_means.append(scores.mean())

plt.figure(figsize=(7,5))
plt.plot(k_vals, tradeoff_means, marker='o')
plt.xlabel("Number of Top Features")
plt.ylabel("CV Balanced Accuracy (RandomForest)")
plt.title("Feature Selection Trade-off")
plt.grid(True)
plt.show()

# Choose top-k for the next stage (adjust if elbow suggests otherwise)
TOP_K = 20
X_topk = X[:, rank_idx[:TOP_K]]

# ---------------------------
# 3) Define base models + randomized search spaces (conditional where needed)
#    Scored by balanced accuracy & F1-macro; refit by balanced accuracy.
# ---------------------------
scoring = {"bal_acc": bal_scorer, "f1_macro": "f1_macro", "accuracy": "accuracy"}

search_spaces = {
    # LDA: conditional solver/shrinkage
    "LDA": (LinearDiscriminantAnalysis(), [
        {"solver": ["svd"], "shrinkage": [None]},
        {"solver": ["lsqr", "eigen"], "shrinkage": ["auto"]},
        {"solver": ["lsqr", "eigen"], "shrinkage": uniform(1e-4, 1.0-1e-4)}  # float in (0,1)
    ]),

    "DT": (DecisionTreeClassifier(random_state=42), {
        "max_depth": randint(3, 20),
        "min_samples_split": randint(2, 30)
    }),

    "ET": (ExtraTreesClassifier(random_state=42), {
        "n_estimators": randint(150, 400),
        "max_depth": randint(5, 30),
        "min_samples_split": randint(2, 30)
    }),

    "GBDT": (GradientBoostingClassifier(random_state=42), {
        "n_estimators": randint(100, 400),
        "learning_rate": uniform(0.01, 0.25),
        "max_depth": randint(2, 8),
        "subsample": uniform(0.6, 0.4)  # 0.6–1.0
    }),

    "NB": (GaussianNB(), {}),  # no hyperparams to tune

    "RF": (RandomForestClassifier(random_state=42), {
        "n_estimators": randint(150, 400),
        "max_depth": randint(5, 30),
        "min_samples_split": randint(2, 30)
    }),

    "KNN": (KNeighborsClassifier(), {
        "n_neighbors": randint(3, 35),
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    }),

    # SVM: conditional kernels
    "SVM": (SVC(probability=True, random_state=42), [
        {"kernel": ["rbf"], "C": reciprocal(1e-3, 1e3), "gamma": reciprocal(1e-4, 1e-1)},
        {"kernel": ["linear"], "C": reciprocal(1e-3, 1e3)},
        {"kernel": ["poly"], "C": reciprocal(1e-3, 1e3),
         "gamma": reciprocal(1e-4, 1e-1), "degree": randint(2, 5)}
    ])
}

# Keep searches moderate so it doesn’t take forever
N_ITER = 20

# ---------------------------
# 4) RandomizedSearchCV for each base model (refit = balanced accuracy)
# ---------------------------
best_models = {}
print("\n=== RandomizedSearchCV (refit=balanced accuracy) ===")
for name, (model, param_dist) in search_spaces.items():
    if isinstance(param_dist, dict) and len(param_dist) == 0:
        # No tuning (e.g., NB)
        best_models[name] = model
        print(f"{name}: no hyperparameters to tune.")
        continue

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,   # can be dict OR list of dicts (conditional)
        n_iter=N_ITER,
        scoring=scoring,
        refit="bal_acc",
        cv=cv,
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_topk, y, groups=groups)
    best_models[name] = search.best_estimator_
    print(f"{name} best params: {search.best_params_} | best CV bal_acc={search.best_score_:.3f}")

# ---------------------------
# 5) Report per-model metrics (Acc, BalAcc, Prec-macro, Rec-macro, F1-macro)
# ---------------------------
def report_metrics(model, X, y, groups):
    y_pred = cross_val_predict(model, X, y, cv=cv, groups=groups, n_jobs=-1)
    acc = accuracy_score(y, y_pred)
    bal = balanced_accuracy_score(y, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average="macro", zero_division=0)
    return {"accuracy": acc, "balanced_accuracy": bal,
            "precision_macro": prec, "recall_macro": rec, "f1_macro": f1}

print("\n=== Tuned Base Models: Cross-validated metrics (top-20 features) ===")
model_reports = {}
for name, mdl in best_models.items():
    metrics_dict = report_metrics(mdl, X_topk, y, groups)
    model_reports[name] = metrics_dict
    print(f"{name:>5s} | Acc={metrics_dict['accuracy']:.3f}  "
          f"BalAcc={metrics_dict['balanced_accuracy']:.3f}  "
          f"PrecM={metrics_dict['precision_macro']:.3f}  "
          f"RecM={metrics_dict['recall_macro']:.3f}  "
          f"F1M={metrics_dict['f1_macro']:.3f}")

# Select top 4 by balanced accuracy
top4 = sorted(model_reports.items(), key=lambda kv: kv[1]["balanced_accuracy"], reverse=True)[:4]
top4_names = [x[0] for x in top4]
print("\nTop-4 models by balanced accuracy:", top4_names)
estimators = [(name, best_models[name]) for name in top4_names]

# ---------------------------
# 6) VotingClassifier: compare hard vs soft + a few weights
# ---------------------------
def eval_voting(voting_type, weights=None):
    vc = VotingClassifier(estimators=estimators, voting=voting_type, weights=weights)
    y_pred = cross_val_predict(vc, X_topk, y, cv=cv, groups=groups, n_jobs=-1)
    acc = accuracy_score(y, y_pred)
    bal = balanced_accuracy_score(y, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average="macro", zero_division=0)
    return {"accuracy": acc, "balanced_accuracy": bal, "precision_macro": prec,
            "recall_macro": rec, "f1_macro": f1}

print("\n=== VotingClassifier comparisons ===")
hard_metrics = eval_voting("hard", None)
print(f"hard voting | Acc={hard_metrics['accuracy']:.3f} "
      f"BalAcc={hard_metrics['balanced_accuracy']:.3f} F1M={hard_metrics['f1_macro']:.3f}")

best_cfg = ("hard", None, hard_metrics["balanced_accuracy"], hard_metrics)
weight_options = [(1,1,1,1), (2,1,1,1), (3,2,1,1), (2,2,1,1), (3,1,1,1)]
for w in weight_options:
    m = eval_voting("soft", w)
    print(f"soft voting, weights={w} | Acc={m['accuracy']:.3f} "
          f"BalAcc={m['balanced_accuracy']:.3f} F1M={m['f1_macro']:.3f}")
    if m["balanced_accuracy"] > best_cfg[2]:
        best_cfg = ("soft", w, m["balanced_accuracy"], m)

final_voting_type, final_weights, _, final_metrics = best_cfg
print(f"\nChosen Voting setup -> type={final_voting_type}, weights={final_weights}")
print(f"Final Voting metrics | Acc={final_metrics['accuracy']:.3f} "
      f"BalAcc={final_metrics['balanced_accuracy']:.3f} "
      f"PrecM={final_metrics['precision_macro']:.3f} "
      f"RecM={final_metrics['recall_macro']:.3f} "
      f"F1M={final_metrics['f1_macro']:.3f}")

final_vc = VotingClassifier(estimators=estimators, voting=final_voting_type, weights=final_weights)
final_vc.fit(X_topk, y)

# ---------------------------
# 7) Learning curve (balanced accuracy) for final VotingClassifier
# ---------------------------
train_sizes, train_scores, val_scores = learning_curve(
    final_vc, X_topk, y, cv=cv, groups=groups,
    train_sizes=np.linspace(0.1, 1.0, 5),
    scoring=bal_scorer, n_jobs=-1
)

plt.figure(figsize=(7,5))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label="Training")
plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label="Validation")
plt.xlabel("Training size")
plt.ylabel("Balanced Accuracy")
plt.title("Learning Curve (Final VotingClassifier)")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------
# 8) Confusion matrix for final VotingClassifier (CV predictions)
# ---------------------------
y_pred_final = cross_val_predict(final_vc, X_topk, y, cv=cv, groups=groups, n_jobs=-1)
cm = confusion_matrix(y, y_pred_final)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot()
plt.title("Confusion Matrix (Final VotingClassifier)")
plt.show()

# ---------------------------
# 9) Summary table of tuned base models (sorted by balanced accuracy)
# ---------------------------
summary_rows = []
for name, m in model_reports.items():
    row = {"model": name}
    row.update(m)
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows).sort_values("balanced_accuracy", ascending=False)
print("\n=== Summary (base models, CV on top-20 features) ===")
print(summary_df.to_string(index=False))
