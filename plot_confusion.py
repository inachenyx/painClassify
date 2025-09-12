import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, cross_val_predict, RandomizedSearchCV, learning_curve


# ---------------------------
# 8. Confusion Matrix
# ---------------------------

df = pd.read_csv('confusion_adjust.csv')

col1_name, col2_name = df.columns[0], df.columns[1]

y = df[col1_name].tolist()
y_pred = df[col2_name].tolist()

custom_labels = ["mild", "moderate", "severe"]
# Row-normalized confusion matrix: each row sums to 1.0
cm = confusion_matrix(y, y_pred, normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=custom_labels)

fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, cmap='Blues', values_format=".2f", colorbar=False)
# plt.colorbar(disp.im_, ax=ax, shrink=0.75)
plt.colorbar(disp.im_, fraction=0.046, pad=0.04)


ax.set_title("Normalized Confusion Matrix")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
# Rotate y-axis labels to be vertical
ax.tick_params(axis='y', rotation=90)
plt.tight_layout()
plt.show()

