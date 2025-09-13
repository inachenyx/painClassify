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
# df = pd.read_csv('confusion_adjust.csv')
# col1_name, col2_name = df.columns[0], df.columns[1]
# y = df[col1_name].tolist()
# y_pred = df[col2_name].tolist()
# custom_labels = ["mild", "moderate", "severe"]
#
# recall = recall_score(y, y_pred, average='macro')
# precision = precision_score(y, y_pred, average='macro')
# f1 = f1_score(y, y_pred, average='macro')
# accuracy = accuracy_score(y, y_pred)
#
# print(f"Recall: {recall}\nPrecision: {precision}\nF1 score: {f1}\nAccuracy: {accuracy}")

# ---------------------------
# Load data from outer training
# ---------------------------
df = pd.read_csv('metrics_adjusted.csv', index_col=0)

# 正确率图

# -------------Top 4 = GB (row=2) here--------------
DTacc_mean= df.iloc[2,0]
DTacc_errors= df.iloc[2,1]

DTf1_mean= df.iloc[2,6]
DTf1_errors= df.iloc[2,7]

DTpre_mean= df.iloc[2,2]
DTpre_errors= df.iloc[2,3]

DTrecall_mean= df.iloc[2,4]
DTrecall_errors= df.iloc[2,5]

# -------------Top 3 = ET (row=1) here--------------
ETacc_mean= df.iloc[1,0]
ETacc_errors= df.iloc[1,1]

ETf1_mean= df.iloc[1,6]
ETf1_errors= df.iloc[1,7]

ETpre_mean= df.iloc[1,2]
ETpre_errors= df.iloc[1,3]

ETrecall_mean= df.iloc[1,4]
ETrecall_errors= df.iloc[1,5]

# -------------Top 2 = RF (row=0) here--------------
GBacc_mean= df.iloc[0,0]
GBacc_errors= df.iloc[0,1]

GBf1_mean= df.iloc[0,6]
GBf1_errors= df.iloc[0,7]

GBpre_mean= df.iloc[0,2]
GBpre_errors= df.iloc[0,3]

GBrecall_mean= df.iloc[0,4]
GBrecall_errors= df.iloc[0,5]

# -------------Top 5 = DT (row=3) here--------------
LDacc_mean= df.iloc[3,0]
LDacc_errors= df.iloc[3,1]

LDf1_mean= df.iloc[3,6]
LDf1_errors= df.iloc[3,7]

LDpre_mean= df.iloc[3,2]
LDpre_errors= df.iloc[3,3]

LDrecall_mean= df.iloc[3,4]
LDrecall_errors= df.iloc[3,5]

# -------------Top 1 = VOTE (row=4) here--------------
voteacc_mean= df.iloc[4,0]
voteacc_errors= df.iloc[4,1]

votef1_mean= df.iloc[4,6]
votef1_errors= df.iloc[4,7]

votepre_mean= df.iloc[4,2]
votepre_errors= df.iloc[4,3]

voterecall_mean= df.iloc[4,4]
voterecall_errors= df.iloc[4,5]

clf_names=['DT',
           'GBDT',
           'ET',
           'RF',
           'VOTE']

# scores=np.array([DT_mean,ET_mean,GB_mean,LD_mean,vote_mean])
# errors=np.array([DT_errors,ET_errors,GB_errors,LD_errors,vote_errors])

f1=np.array([LDf1_mean,DTf1_mean,ETf1_mean,GBf1_mean,votef1_mean])
f1errors=np.array([LDf1_errors,DTf1_errors,ETf1_errors,GBf1_errors,votef1_errors])

pre=np.array([LDpre_mean,DTpre_mean,ETpre_mean,GBpre_mean,votepre_mean])
preerrors=np.array([LDpre_errors,DTpre_errors,ETpre_errors,GBpre_errors,votepre_errors])

recall=np.array([LDrecall_mean,DTrecall_mean,ETrecall_mean,GBrecall_mean,voterecall_mean])
recallerrors=np.array([LDrecall_errors,DTrecall_errors,ETrecall_errors,GBrecall_errors,voterecall_errors])

acc=np.array([LDacc_mean,DTacc_mean,ETacc_mean,GBacc_mean,voteacc_mean])
accerrors=np.array([LDacc_errors,DTacc_errors,ETacc_errors,GBacc_errors,voteacc_errors])


fenleiqi=np.array(["DT","GBDT", "ET", "RF","VOTE"])
x_len = np.arange(len(fenleiqi))
print(x_len)
total_width, n = 0.9, 4
width = 0.2
xticks = x_len - (total_width - width) / 2
#plt.bar(fenleiqi,scores,width=0.3,yerr=errors,error_kw = {'ecolor' : '0.2', 'capsize' :6 }, alpha=0.7)
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.direction'] = 'in'
font = {'family': 'Arial'  # 'serif',
        #         ,'style':'italic'
    , 'weight': 'bold'  # 'normal'
        #         ,'color':'red'
    , 'size': 16
        }
fig, ax = plt.subplots(nrows=1, ncols=1,
                       figsize=(14, 7),
                       dpi=100, facecolor="w")
plt.bar(xticks, f1, width=0.9*width,color="#6788ED",label="OriginalFeature",yerr = f1errors,error_kw = {'ecolor' : '0.2', 'capsize' :6 }, alpha=0.8)
plt.bar(xticks+width, pre, width=0.9*width,color="#99BAFE",label="SelectFeature",yerr = preerrors,error_kw = {'ecolor' : '0.2', 'capsize' :6 }, alpha=0.8)
plt.bar(xticks+2*width, recall, width=0.9*width,color="#F6A789",label="SelectFeature",yerr =recallerrors,error_kw = {'ecolor' : '0.2', 'capsize' :6 }, alpha=0.8)
plt.bar(xticks+3*width, acc, width=0.9*width,color="#E16852",label="SelectFeature",yerr = accerrors,error_kw = {'ecolor' : '0.2', 'capsize' :6 }, alpha=0.8)
plt.xticks(x_len,fenleiqi, fontproperties=font)

# for m, n ,z in zip(fenleiqi, scores,errors):
#     # ha: horizontal alignment
#     # va: vertical alignment
#     plt.text(m, n +z, '%.2f' %n, ha='center', va='bottom')
# num2=1表示legend位于图像上侧水平线(其它设置：num1=1.05,num3=3,num4=0)。
num1 = 2
num2 = 1.2
num3 = 3
num4 = 0
plt.legend(prop={'family' : 'Arial', 'size': 14,'weight': 'bold'}, labels=['F1','Precision','Recall','Accuracy'],ncol=4,bbox_to_anchor=(0,1.01,1.0,2), loc=8)

for m, n ,z in zip(xticks, f1,f1errors):
    # ha: horizontal alignment
    # va: vertical alignment
    plt.text(m, n +z, '%.2f' %n, ha='center', va='bottom',fontdict = font)
for m, n ,z in zip(xticks, pre,preerrors):
    # ha: horizontal alignment
    # va: vertical alignment
    plt.text(m+width, n +z, '%.2f' %n, ha='center', va='bottom',fontdict = font)
for m, n ,z in zip(xticks, recall,recallerrors):
    # ha: horizontal alignment
    # va: vertical alignment
    plt.text(m+2*width, n +z, '%.2f' %n, ha='center', va='bottom',fontdict = font)
for m, n ,z in zip(xticks, acc,accerrors):
    # ha: horizontal alignment
    # va: vertical alignment
    plt.text(m+3*width, n +z, '%.2f' %n, ha='center', va='bottom',fontdict = font)

plt.xlabel("Classifiers",fontdict = font)
plt.ylabel("Scores",fontdict = font)
# plt.title('Accuracy Rate',fontdict = font)
plt.ylim(0, 1)
ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontproperties=font)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.show()





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

