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
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, cross_val_predict, RandomizedSearchCV, learning_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import randint, uniform

import seaborn as sns
from statannotations.Annotator import Annotator

# ---------------------------
# 1. Load and prepare data
# ---------------------------
mat = sio.loadmat("allfeature.mat")
data = mat["allfeature"]

# feature names
features = ['max','min','mean','med','peak','arv','var','std','kurtosis','skewness','rms',
            'rs','rmsa','waveformF','peakF','impulseF','clearanceF','FC','MSF','RMSF','VF',
            'RVF','SKMean','SKStd','SKSkewness','SKKurtosis','psdE','svdpE','eE','ApEn',
            'SpEn','FuzzyEn','PeEn','enveEn','detaDE','thetaDE','alphaDE','betaDE','gammaDE']
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

# Violin plot with wilcoxon test
forviolin_tick=['alphaDE_ch1','betaDE_ch1','gammaDE_ch1','PeEn_ch1','alphaDE_ch2','betaDE_ch2','gammaDE_ch2','PeEn_ch2']

def forviolin(ax,df,feature):
    ax = sns.violinplot(x=df["label"], y=df[feature],palette=['#99BAFE','#F6A789','#E16852'],ax=ax)
    pairs = [("mild", "moderate"), ("moderate", "severe"), ("mild", "severe")]
    annotator = Annotator(ax, pairs, x=df["label"], y=df[feature], )
    annotator.configure(test='Mann-Whitney', text_format='star', line_height=0.03, line_width=1)
    annotator.apply_and_annotate()

    ax.tick_params(which='major', direction='in', length=3, width=1., labelsize=12, bottom=False)
    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.grid(axis='y', ls='--', c='gray')
    ax.set_axisbelow(True)
    font = {'family':'serif', 'weight':'bold' ,'color':'black', 'size':10}
    ax.set_xlabel('class', fontdict = font)
    ax.set_ylabel(feature, fontdict = font)
    return ax

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 6), dpi=100, facecolor="w")
for j in range(0, 2):
    for i in range(0, 4):
        forviolin(ax[j, i], df, forviolin_tick[4*j+i])
plt.show()