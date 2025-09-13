import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 14
font_dict={'fontname': "Arial", 'fontsize': 14, 'fontweight': 'bold'}

df = pd.read_csv('learning_adjust.csv')

train_sizes = df['train_size']
train_mean = df['train_mean']
train_std = df['train_std']
val_mean = df['cv_mean']
val_std = df['cv_std']


fig, ax = plt.subplots(figsize=(6, 4.28))

line1, = ax.plot(train_sizes, train_mean, marker="o", label="Training score", color="#6788ED")
line2, = ax.plot(train_sizes, val_mean, marker="o", label="Cross-validation score", color="#E16852")

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

ax.set_xlabel("Training size", fontdict=font_dict)
ax.set_ylabel("Accuracy", fontdict=font_dict)
ax.set_title("Voting Classifier Learning Curve", fontdict=font_dict)
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()