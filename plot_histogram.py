import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, cross_val_predict, RandomizedSearchCV, learning_curve

# ---------------------------
# Calculate metrics on outer testing
# ---------------------------
df = pd.read_csv('confusion_adjust.csv')
col1_name, col2_name = df.columns[0], df.columns[1]
y = df[col1_name].tolist()
y_pred = df[col2_name].tolist()
custom_labels = ["mild", "moderate", "severe"]

recall = recall_score(y, y_pred, average='macro')
precision = precision_score(y, y_pred, average='macro')
f1 = f1_score(y, y_pred, average='macro')
accuracy = accuracy_score(y, y_pred)

print(f"Recall: {recall}\nPrecision: {precision}\nF1 score: {f1}\nAccuracy: {accuracy}")

# ---------------------------
# Load data from outer training
# ---------------------------

# # ---------------------------
# # 9. Histogram plot
# # ---------------------------
# models_for_plot = top4_names + ["Voting"]
# name_to_est = {name: best_models[name] for name in top4_names}
# name_to_est["Voting"] = final_vc
#
# scorers = [
#     ("accuracy", "accuracy"),
#     ("precision_macro", "precision_macro"),
#     ("recall_macro", "recall_macro"),
#     ("f1_macro", "f1_macro"),
# ]
#
# means = []
# stds  = []
# for m in models_for_plot:
#     est = name_to_est[m]
#     m_means = []
#     m_stds  = []
#     for label, scoring in scorers:
#         sc = cross_val_score(est, X_top20, y, cv=cv, groups=groups,
#                              scoring=scoring, n_jobs=-1)
#         m_means.append(np.nanmean(sc))
#         m_stds.append(np.nanstd(sc))
#     means.append(m_means)
#     stds.append(m_stds)
#
# means = np.array(means)  # shape: (n_models, 4)
# stds  = np.array(stds)
#
# plt.figure(figsize=(9, 5))
# n_models = len(models_for_plot)
# n_metrics = len(scorers)
# x = np.arange(n_models)
# width = 0.18
#
# for j, (label, _) in enumerate(scorers):
#     plt.bar(x + (j - (n_metrics-1)/2)*width, means[:, j], width=width,
#             yerr=stds[:, j], capsize=3, label=label)
#
# plt.xticks(x, models_for_plot, rotation=0)
# plt.ylabel("Score")
# plt.title("CV metrics (mean ± std)")
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# # columns: model, metric_mean, metric_std (wide format)
# cols = []
# for label, _ in [("accuracy","accuracy"), ("precision_macro","precision_macro"),
#                  ("recall_macro","recall_macro"), ("f1_macro","f1_macro")]:
#     cols.extend([f"{label}_mean", f"{label}_std"])
#
# df_metrics = pd.DataFrame(columns=["model"] + cols)
# for i, m in enumerate(models_for_plot):
#     row = {"model": m}
#     j = 0
#     for label, _ in [("accuracy","accuracy"), ("precision_macro","precision_macro"),
#                      ("recall_macro","recall_macro"), ("f1_macro","f1_macro")]:
#         row[f"{label}_mean"] = float(means[i, j])
#         row[f"{label}_std"]  = float(stds[i, j])
#         j += 1
#     df_metrics.loc[len(df_metrics)] = row
#
# df_metrics.to_csv("metrics_histogram.csv", index=False)
# print("[saved] metrics_histogram.csv")