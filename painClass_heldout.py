import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, VotingClassifier)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, RandomizedSearchCV, learning_curve
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from scipy.stats import randint, uniform

"""
Dataset: 20 subjects each with 48 segments of data (960 segments total). Each subject belongs to one level of pain.
Labels: mild, moderate, severe pain level (All segments from one subject have the same labels)
Features: 78 features were extracted (39 different types of feature * 2 EEG channels Fp1 & Fp2)
The top 20 performing features were selected for model training.
Models: A variety of models were trained and tuned for best hyperparameters. 
The top 4 models were used to train a Voting Classifier.
StratifiedKFold cv was used for all nested cross-validations. 
"""
plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12

rndst = 20
# ---------------------------
# 1. Load and prepare data
# ---------------------------
mat = sio.loadmat("allfeature.mat")
data = mat["allfeature"]

# feature names
features = ['max','min','mean','med','peak','arv','var','std','kurtosis','skewness','rms',
            'rs','rmsa','waveformF','peakF','impulseF','clearanceF','FC','MSF','RMSF','VF',
            'RVF','SKMean','SKStd','SKSkewness','SKKurtosis','psdE','svdpE','eE','ApEn',
            'SpEn','FuzzyEn','PeEn','enveEn','deltaDE','thetaDE','alphaDE','betaDE','gammaDE']
featuremap = {i: features[i] + "_ch1" if i <= 38 else features[i-39] + "_ch2" for i in range(78)}
feature_names = [featuremap[i] for i in range(78)]

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

# Encode labels
X = df.iloc[:, :-1].values
y_str = df["label"].values
le = LabelEncoder()
y = le.fit_transform(y_str)   # mild=0, moderate=1, severe=2
groups = np.repeat(np.arange(20), 48)   # subject IDs

# ---------------------------
# ===== Hold out 3 subjects for test: one subject per class (0,1,2) =====
# ---------------------------
HOLDOUT_RANDOM_STATE = rndst
# Optionally hard-pick subjects per class, e.g. {class: subject_id}
HOLDOUT_SUBJECT_IDS = None  # e.g., {0: 3, 1: 12, 2: 17}

rng = np.random.RandomState(HOLDOUT_RANDOM_STATE)

# Derive a single label per subject (assumes each subject has one label)
subject_ids = np.unique(groups) # [0...19]
subject_label = {} # dict{sid:label} = {0:0, 1:0, ..., 6:0, 7:1, ..., 19:2}
for sid in subject_ids:
    labels_sid = y[groups == sid] # the corresponding 48 labels in y
    maj = np.bincount(labels_sid).argmax() # find the mode
    subject_label[sid] = maj

# Select one subject per class
if isinstance(HOLDOUT_SUBJECT_IDS, dict):
    test_subjects = np.array([HOLDOUT_SUBJECT_IDS[c] for c in [0,1,2]])
else:
    test_subjects = []
    for c in [0, 1, 2]:
        cands = [sid for sid, lab in subject_label.items() if lab == c]
        if not cands:
            raise ValueError(f"No subjects found for class {c}")
        test_subjects.append(rng.choice(cands))
    test_subjects = np.array(test_subjects)

mask_test  = np.isin(groups, test_subjects)
mask_train = ~mask_test

X_train, y_train, groups_train = X[mask_train], y[mask_train], groups[mask_train]
X_test,  y_test,  groups_test  = X[mask_test],  y[mask_test],  groups[mask_test]

print("Held-out subject IDs (per class 0/1/2):", test_subjects)
print("Train size:", X_train.shape, " Test size:", X_test.shape)
print(f"Test subjects: {sorted(test_subjects.tolist())}")

# CV objects
# inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rndst)  # for tuning and model ranking on TRAIN only


# ---------------------------
# Define CV here (prevents one-class folds during CV scoring)
# ---------------------------
def _safe_splits_for_groups(y, groups, requested):
    # number of unique subjects in each class
    counts = [len(np.unique(groups[y == c])) for c in np.unique(y)]
    max_splits = max(2, min(counts))  # at least 2
    if requested > max_splits:
        print(f"[note] Reducing n_splits from {requested} to {max_splits} "
              f"because some classes have only {max_splits} subject-groups.")
        return max_splits
    return requested

requested_splits = 5  # whatever you currently use
n_splits_safe = _safe_splits_for_groups(y, groups, requested_splits)
cv = StratifiedGroupKFold(n_splits=n_splits_safe, shuffle=True, random_state=rndst)

# ---------------------------
# 2.1 Feature importance + trade-off plot
# ---------------------------
# rf = RandomForestClassifier(random_state=rndst, n_estimators=200)
# rf.fit(X, y)
# importances = rf.feature_importances_
# indices = np.argsort(importances)[::-1]
# sorted_features = [feature_names[i] for i in indices]
#
# mean_scores = []
# for k in range(5, 79, 5):
#     X_topk = X[:, indices[:k]]
#     scores = cross_val_score(rf, X_topk, y, cv=cv, groups=groups, scoring="accuracy")
#     mean_scores.append(scores.mean())
#
# plt.figure(figsize=(7,5))
# plt.plot(range(5,79,5), mean_scores, marker="o")
# plt.xlabel("Number of Top Features")
# plt.ylabel("CV Accuracy (RF)")
# plt.title("Feature Selection Trade-off")
# plt.grid(True)
# plt.show()

# ---------------------------
# 2.2 ET Top-20 feature importance with ExtraTrees
# ---------------------------
et_tmp = ExtraTreesClassifier(n_estimators=400, random_state=rndst, n_jobs=-1)
et_tmp.fit(X_train, y_train)
importances = et_tmp.feature_importances_
idx = np.argsort(importances)[::-1]
sorted_f = [feature_names[i] for i in idx]

mean_scores = []
for k in range(5, 79, 5):
    X_topk = X[:, idx[:k]]
    scores = cross_val_score(et_tmp, X_topk, y, cv=cv, groups=groups, scoring="accuracy")
    mean_scores.append(scores.mean())

plt.figure(figsize=(7,5))
plt.plot(range(5,79,5), mean_scores, marker="o")
plt.xlabel("Number of Top Features")
plt.ylabel("CV Accuracy (EF)")
plt.title("Feature Selection Trade-off")
plt.grid(True)
plt.show()

# Choose top-20 features for next step
X_top20 = X_train[:, idx[:20]]
X_test_top20 = X_test[:, idx[:20]]

# ---------------------------
# 3. Define models + hyperparam spaces
# ---------------------------
search_spaces = {
    "LDA": (LinearDiscriminantAnalysis(), {
        "solver": ["svd", "lsqr", "eigen"],
        "n_components": [1,2,None],
        "tol": np.linspace(0.0001, 0.001, 10)
    }),
    "DT": (DecisionTreeClassifier(random_state=rndst), {
        "max_depth": [3,15,100,None],
        "max_features": randint(1, 11),
        "min_samples_split": randint(2, 11),
        "min_samples_leaf": randint(4, 11),
        "criterion": ["gini", "entropy"]
    }),
    "ET": (ExtraTreesClassifier(random_state=rndst), {
        "n_estimators": [10,50,100,200],
        "max_depth": [5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False]
    }),
    "GBDT": (GradientBoostingClassifier(random_state=rndst), {
        "n_estimators": range(30,150),
        "learning_rate": np.arange(0.01, 0.5, 0.01),
        "max_depth": range(2,7),
        "max_features": range(2,9)
    }),
    "NB": (GaussianNB(), {}),
    "RF": (RandomForestClassifier(random_state=rndst), {
        "n_estimators": randint(100, 500),
        "max_depth": randint(5, 30),
        "min_samples_split": randint(2, 20)
    }),
    "KNN": (KNeighborsClassifier(), {
        "n_neighbors": randint(3, 30),
        "weights": ["uniform", "distance"]
    }),
    # "SVM": (Pipeline([
    #     ("scaler", StandardScaler()),
    #     ("svc", SVC(probability=True, random_state=rndst))
    # ]), {
    #     "svc__kernel": ["rbf", "linear"],
    #     "svc__C": uniform(0.1, 100.0),
    #     "svc__gamma": ["scale", "auto"]   # used for 'rbf'; ignored for 'linear'
    # })
}

# ---------------------------
# 4. RandomizedSearchCV tuning
# ---------------------------
best_models = {}
for name, (model, param_dist) in search_spaces.items():
    if len(param_dist) == 0:
        best_models[name] = model
        continue
    search = RandomizedSearchCV(
        model, param_distributions=param_dist,
        n_iter=25, cv=cv, scoring="accuracy", n_jobs=-1, random_state=rndst,
        refit = True, verbose = 0
    )
    search.fit(X_top20, y_train, groups=groups_train)
    best_models[name] = search.best_estimator_
    print(f"{name} best params:", search.best_params_)

# ---------------------------
# 5. Evaluate tuned models
# ---------------------------
results = {} # mean
scores_summary = {} # mean, std
for name, model in best_models.items():
    scores = cross_val_score(model, X_top20, y_train, cv=cv, groups=groups_train, scoring="accuracy")
    scores_summary[name] = (scores.mean(), scores.std())
    print(f"{name:>5}: mean={scores.mean():.3f} ± {scores.std():.3f}")
    results[name] = scores.mean()

sorted_models = sorted(results.items(), key=lambda x: x[1], reverse=True)
top4_names = [m[0] for m in sorted_models[:4]]
print("Top 4 tuned models:", top4_names)

estimators = [(name, best_models[name]) for name in top4_names]

# ---------------------------
# 6. VotingClassifier: hard vs soft + weights
# ---------------------------
for voting_type in ["hard", "soft"]:
    if voting_type == "soft":
        for weights in [(1,1,1,1), (2,1,1,1), (3,2,1,1)]:
            vc = VotingClassifier(estimators=estimators, voting=voting_type, weights=weights)
            score = cross_val_score(vc, X_top20, y_train, cv=cv, groups=groups_train, scoring="accuracy").mean()
            print(f"Voting={voting_type}, weights={weights}, acc={score:.3f}")
    else:
        vc = VotingClassifier(estimators=estimators, voting=voting_type)
        score = cross_val_score(vc, X_top20, y_train, cv=cv, groups=groups_train, scoring="accuracy").mean()
        print(f"Voting={voting_type}, acc={score:.3f}")

# Choose a final VotingClassifier (example: soft voting, weighted)
final_vc = VotingClassifier(estimators=estimators, voting="soft", weights=(2,1,1,1))
final_vc.fit(X_top20, y_train)

# ---------------------------
# 7. Learning curve
# ---------------------------
def balanced_prefix_cv(cv, X, y, groups, min_classes=2, front_subjects_by_class=None):
    """
    Wrap a group-aware splitter so that, for each split, the *order* of the
    training indices begins with whole-group blocks from at least `min_classes`
    distinct classes. This makes small train_size prefixes (e.g., 96) multi-class.

    If you want manual control, pass front_subjects_by_class={0: sid0, 1: sid1, ...}.
    """
    for train_idx, test_idx in cv.split(X, y, groups):
        tr_g = groups[train_idx]
        tr_y = y[train_idx]

        front_order = []
        used_groups = set()

        # Optional manual seeding: bring chosen subjects to the front
        if isinstance(front_subjects_by_class, dict):
            for c, sid in front_subjects_by_class.items():
                mask = (tr_g == sid)
                if np.any(mask):
                    front_order.extend(np.where(mask)[0].tolist())
                    used_groups.add(sid)

        # Ensure at least `min_classes` distinct-class groups at the front
        if len(used_groups) < min_classes:
            classes = np.unique(tr_y)
            for c in classes:
                if len(used_groups) >= min_classes:
                    break
                # pick one not-yet-used group from class c
                c_grp_ids = np.unique(tr_g[tr_y == c])
                for gid in c_grp_ids:
                    if gid not in used_groups:
                        mask = (tr_g == gid)
                        front_order.extend(np.where(mask)[0].tolist())
                        used_groups.add(gid)
                        break

        # Append the rest (any order is fine)
        all_idx = set(range(len(train_idx)))
        rest = [i for i in all_idx if i not in set(front_order)]
        perm = np.array(front_order + rest, dtype=int)

        # Reorder the train indices for this split
        yield train_idx[perm], test_idx

# Keep your group-aware CV "cv" for everything else (tuning, scoring, etc.)
# A CV iterable for the learning curve that guarantees multi-class prefixes
lc_cv = list(balanced_prefix_cv(cv, X_top20, y_train, groups_train, min_classes=2))
# If you want to pin exactly which subjects appear first at the smallest size:
# lc_cv = list(balanced_prefix_cv(cv, X_top20, y, groups, min_classes=2,
#                                 front_subjects_by_class={0: 3, 1: 7, 2: 14}))

train_sizes, train_scores, val_scores = learning_curve(
    final_vc, X_top20, y_train, cv=lc_cv,
    # groups=groups,
    train_sizes=np.linspace(0.1, 1.0, 5), scoring="accuracy", n_jobs=-1,
    shuffle=False, random_state=rndst
)

train_mean = np.nanmean(train_scores, axis=1)
val_mean = np.nanmean(val_scores, axis=1)

train_std = np.nanstd(train_scores, axis=1)
val_std = np.nanstd(val_scores, axis=1)

fig, ax = plt.subplots(figsize=(7, 5))

line1, = ax.plot(train_sizes, train_mean, marker="o", label="Training score", color="#6788ED")
line2, = ax.plot(train_sizes, val_mean, marker="s", label="Cross-validation score", color="#E16852")

# Shaded ±1 std around each curve
ax.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.15,
    edgecolor="none",
    facecolor=line1.get_color()
)
ax.fill_between(
    train_sizes,
    val_mean - val_std,
    val_mean + val_std,
    alpha=0.15,
    edgecolor="none",
    facecolor=line2.get_color()
)

ax.set_xlabel("Training size")
ax.set_ylabel("Accuracy")
ax.set_title("Voting Classifier Learning Curve")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

# plt.figure(figsize=(7,5))
# plt.plot(train_sizes, train_mean, 'o-', label="Training")
# plt.plot(train_sizes, val_mean, 'o-', label="Validation")
# plt.xlabel("Training size")
# plt.ylabel("Accuracy")
# plt.title("Learning Curve (VotingClassifier)")
# plt.legend()
# plt.grid(True)
# plt.show()

# ---------------------------
# 8. Confusion Matrix
# ---------------------------
y_test_pred = final_vc.predict(X_test_top20)
print("\n=== Held-out test classification report (Voting) ===")
print(classification_report(y_test, y_test_pred))
# y_pred = cross_val_predict(final_vc, X_top20, y, cv=cv, groups=groups)
# Row-normalized confusion matrix: each row sums to 1.0
cm = confusion_matrix(y_test, y_test_pred, normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, cmap='Blues', values_format=".2f", colorbar=False)
plt.colorbar(disp.im_, fraction=0.046, pad=0.04)

ax.set_title("Normalized Confusion Matrix")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
# Rotate y-axis labels to be vertical
ax.tick_params(axis='y', rotation=90)
plt.tight_layout()
plt.show()



