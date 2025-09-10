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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
# 2.1 Feature importance + trade-off plot
# ---------------------------
rf = RandomForestClassifier(random_state=42, n_estimators=200)
rf.fit(X, y)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = [feature_names[i] for i in indices]

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
mean_scores = []
for k in range(5, 79, 5):
    X_topk = X[:, indices[:k]]
    scores = cross_val_score(rf, X_topk, y, cv=cv, groups=groups, scoring="accuracy")
    mean_scores.append(scores.mean())

plt.figure(figsize=(7,5))
plt.plot(range(5,79,5), mean_scores, marker="o")
plt.xlabel("Number of Top Features")
plt.ylabel("CV Accuracy (RF)")
plt.title("Feature Selection Trade-off")
plt.grid(True)
plt.show()

# ---------------------------
# 2.2 ET Top-20 feature importance with ExtraTrees
# ---------------------------
et_tmp = ExtraTreesClassifier(n_estimators=400, random_state=42, n_jobs=-1)
et_tmp.fit(X, y)
importances = et_tmp.feature_importances_
idx = np.argsort(importances)[::-1]
sorted_f = [feature_names[i] for i in idx]

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
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
X_top20 = X[:, indices[:20]]

# ---------------------------
# 3. Define models + hyperparam spaces
# ---------------------------
search_spaces = {
    "LDA": (LinearDiscriminantAnalysis(), {
        "solver": ["svd", "lsqr", "eigen"],
        "shrinkage": [None, "auto"]
    }),
    "DT": (DecisionTreeClassifier(random_state=42), {
        "max_depth": randint(3, 20),
        "min_samples_split": randint(2, 20)
    }),
    "ET": (ExtraTreesClassifier(random_state=42), {
        "n_estimators": randint(100, 500),
        "max_depth": randint(5, 30),
        "min_samples_split": randint(2, 20)
    }),
    "GBDT": (GradientBoostingClassifier(random_state=42), {
        "n_estimators": randint(100, 500),
        "learning_rate": uniform(0.01, 0.3),
        "max_depth": randint(2, 10)
    }),
    "NB": (GaussianNB(), {}),
    "RF": (RandomForestClassifier(random_state=42), {
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
    #     ("svc", SVC(probability=True, random_state=42))
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
        n_iter=20, cv=cv, scoring="accuracy", n_jobs=-1, random_state=42
    )
    search.fit(X_top20, y, groups=groups)
    best_models[name] = search.best_estimator_
    print(f"{name} best params:", search.best_params_)

# ---------------------------
# 5. Evaluate tuned models
# ---------------------------
results = {} # mean
scores_summary = {} # mean, std
for name, model in best_models.items():
    scores = cross_val_score(model, X_top20, y, cv=cv, groups=groups, scoring="accuracy")
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
            score = cross_val_score(vc, X_top20, y, cv=cv, groups=groups, scoring="accuracy").mean()
            print(f"Voting={voting_type}, weights={weights}, acc={score:.3f}")
    else:
        vc = VotingClassifier(estimators=estimators, voting=voting_type)
        score = cross_val_score(vc, X_top20, y, cv=cv, groups=groups, scoring="accuracy").mean()
        print(f"Voting={voting_type}, acc={score:.3f}")

# Choose a final VotingClassifier (example: soft voting, weighted)
final_vc = VotingClassifier(estimators=estimators, voting="soft", weights=(2,1,1,1))
final_vc.fit(X_top20, y)

# ---------------------------
# 7. Learning curve
# ---------------------------
train_sizes, train_scores, val_scores = learning_curve(
    final_vc, X_top20, y, cv=cv, groups=groups,
    train_sizes=np.linspace(0.1, 1.0, 5), scoring="accuracy", n_jobs=-1
)

plt.figure(figsize=(7,5))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label="Training")
plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label="Validation")
plt.xlabel("Training size")
plt.ylabel("Accuracy")
plt.title("Learning Curve (VotingClassifier)")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------
# 8. Confusion Matrix
# ---------------------------
y_pred = cross_val_predict(final_vc, X_top20, y, cv=cv, groups=groups)
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues")
plt.show()

