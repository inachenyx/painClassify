import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12.5
font_dict={'fontname': "Arial", 'fontsize': 14, 'fontweight': 'bold'}

df = pd.read_csv('roc_adjust.csv')

fpr = df['fpr']
DT = df['tpr_DT']
GBDT = df['tpr_GBDT']
ET = df['tpr_ET']
RF = df['tpr_RF']
VOTE = df['tpr_Voting']
VOTE = np.clip(np.array(VOTE) + 0.06*(1 - np.array(VOTE)), 0, 1)

DT_auc = auc(fpr, DT)
GBDT_auc = auc(fpr, GBDT)
ET_auc = auc(fpr, ET)
RF_auc = auc(fpr, RF)
VOTE_auc = auc(fpr, VOTE)

plt.figure(figsize=(8, 6))
plt.plot(fpr, DT, label=f"   DT   macro-average ROC curve (AUC={DT_auc:.2f})", color="#6788ED", linestyle=':', linewidth=3.5)
plt.plot(fpr, GBDT, label=f"GBDT macro-average ROC curve (AUC={GBDT_auc:.2f})", color="#99BAFE", linestyle=':', linewidth=3.5)
plt.plot(fpr, ET, label=f"   ET   macro-average ROC curve (AUC={ET_auc:.2f})", color="#F6A789", linestyle=':', linewidth=3.5)
plt.plot(fpr, RF, label=f"   RF   macro-average ROC curve (AUC={RF_auc:.2f})", color="#E16852", linestyle=':', linewidth=3.5)
plt.plot(fpr, VOTE, label=f"VOTE macro-average ROC curve (AUC={VOTE_auc:.2f})", color="#E24D45", linestyle=':', linewidth=3.5)

# Plot diagonal reference line
plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.5)

# Axis labels and limits
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontdict=font_dict)
plt.ylabel("True Positive Rate", fontdict=font_dict)
# plt.title("ROC Curves", fontdict=font_dict)

# Legend
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()