import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from statannotations.Annotator import Annotator
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests

# ---------------------------
# 1. Load and prepare data
# ---------------------------
mat = sio.loadmat("allfeature.mat")
data = mat["allfeature"]

# Feature names
features = ['max','min','mean','med','peak','arv','var','std','kurtosis','skewness','rms',
            'rs','rmsa','waveformF','peakF','impulseF','clearanceF','FC','MSF','RMSF','VF',
            'RVF','SKMean','SKStd','SKSkewness','SKKurtosis','psdE','svdpE','eE','ApEn',
            'SpEn','FuzzyEn','PeEn','enveEn','detaDE','thetaDE','alphaDE','betaDE','gammaDE']
featuremap = {i: features[i] + "_ch1" if i <= 38 else features[i-39] + "_ch2" for i in range(78)}
feature_names = [featuremap[i] for i in range(78)]

df = pd.DataFrame(data, columns=feature_names)

# Labels: 20 subjects × 48 segments = 960
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
y_str = df["label"].values
le = LabelEncoder()
y = le.fit_transform(y_str)   # mild=0, moderate=1, severe=2

# ---------------------------
# 2. Features of interest
# ---------------------------
forviolin_tick = [
    'alphaDE_ch1','betaDE_ch1','gammaDE_ch1','PeEn_ch1',
    'alphaDE_ch2','betaDE_ch2','gammaDE_ch2','PeEn_ch2'
]

# ---------------------------
# 3. Wilcoxon tests
# ---------------------------
comparisons = [("mild","moderate"),("moderate","severe"),("mild","severe")]
results = []

for feat in forviolin_tick:
    for g1,g2 in comparisons:
        stat,pval = ranksums(df.loc[df["label"]==g1, feat],
                             df.loc[df["label"]==g2, feat])
        results.append({
            "feature": feat,
            "comparison": f"{g1} vs {g2}",
            "pval_raw": pval
        })

resdf = pd.DataFrame(results)

# Apply corrections across all tests
_, p_bonf, _, _ = multipletests(resdf["pval_raw"], method="bonferroni")
_, p_fdr, _, _ = multipletests(resdf["pval_raw"], method="fdr_bh")
resdf["pval_bonf"] = p_bonf
resdf["pval_fdr"] = p_fdr

resdf.to_csv("wilcoxon_multiple_corrections.csv", index=False)

# ---------------------------
# 4A. Violin plot with FDR stars (replacement for manuscript)
# ---------------------------
def violin_with_annotations(feature, ax, pvals):
    ax = sns.violinplot(x=df["label"], y=df[feature],
                        palette=['#99BAFE','#F6A789','#E16852'], ax=ax)
    pairs = [("mild","moderate"),("moderate","severe"),("mild","severe")]
    annotator = Annotator(ax, pairs, x=df["label"], y=df[feature])
    annotator.set_pvalues_and_annotate(list(pvals))
    ax.tick_params(labelsize=16)
    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.grid(axis='y', ls='--', c='gray')
    ax.set_axisbelow(True)
    font = {'family': 'arial', 'weight': 'bold', 'color': 'black', 'fontsize': 18}
    ax.set_xlabel('class', fontdict=font)
    ax.set_ylabel(feature, fontdict=font)
    return ax

plt.rcParams['font.family'] = 'Arial'
# ---------------------------
# Full 8-feature FDR panel
#
# Build a dict of FDR p-values for each feature
fdr_map = {}
for feat in forviolin_tick:
    fdr_map[feat] = resdf.loc[resdf["feature"]==feat, "pval_fdr"].values

fig, axes = plt.subplots(2,4, figsize=(18,8))
axes = axes.ravel()
for i,feat in enumerate(forviolin_tick):
    pvals = fdr_map[feat]
    violin_with_annotations(feat, axes[i], pvals)
# Add numbering to each subplot
axes_flat = axes.flatten()
subnum = ['a','b','c','d','e','f','g','h']
for n, ax in zip(subnum, axes_flat):
    ax.text(-0.22, 1, f'({n})', transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='top', ha='left')
plt.tight_layout()
plt.savefig("violin_FDR.png", dpi=300)
plt.show()
# ---------------------------
# Full 8-feature Bonferroni panel
# ---------------------------
bonf_map = {}
for feat in forviolin_tick:
    bonf_map[feat] = resdf.loc[resdf["feature"]==feat, "pval_bonf"].values

fig, axes = plt.subplots(2,4, figsize=(18,7))
axes = axes.ravel()
for i,feat in enumerate(forviolin_tick):
    pvals = bonf_map[feat]
    violin_with_annotations(feat, axes[i], pvals)
# Add numbering to each subplot
axes_flat = axes.flatten()
subnum = ['a','b','c','d','e','f','g','h']
for n, ax in zip(subnum, axes_flat):
    ax.text(-0.2, 1, f'({n})', transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='top', ha='left')
plt.tight_layout()
plt.savefig("violin_Bonferroni.png", dpi=300)
plt.show()
# ---------------------------
# Full 8-feature raw panel
# ---------------------------
raw_map = {}
for feat in forviolin_tick:
    raw_map[feat] = resdf.loc[resdf["feature"]==feat, "pval_raw"].values

fig, axes = plt.subplots(2,4, figsize=(18,7))
axes = axes.ravel()
for i,feat in enumerate(forviolin_tick):
    pvals = raw_map[feat]
    violin_with_annotations(feat, axes[i], pvals)
# Add numbering to each subplot
axes_flat = axes.flatten()
subnum = ['a','b','c','d','e','f','g','h']
for n, ax in zip(subnum, axes_flat):
    ax.text(-0.2, 1, f'({n})', transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='top', ha='left')
plt.tight_layout()
plt.savefig("violin_raw.png", dpi=300)
plt.show()
# # ---------------------------
# # 4B. Side-by-side raw vs corrected
# # ---------------------------
# for feat in forviolin_tick:
#     fig, axs = plt.subplots(1,3, figsize=(15,5))
#     for j,(label, col) in enumerate(zip(["Raw","Bonferroni","FDR"],
#                                         ["pval_raw","pval_bonf","pval_fdr"])):
#         ax = sns.violinplot(x=df["label"], y=df[feat],
#                             palette=['#99BAFE','#F6A789','#E16852'], ax=axs[j])
#         pairs = [("mild","moderate"),("moderate","severe"),("mild","severe")]
#         annotator = Annotator(ax, pairs, x=df["label"], y=df[feat])
#         annotator.set_pvalues_and_annotate(list(resdf.loc[resdf["feature"]==feat, col]))
#         ax.set_title(f"{feat} ({label})")
#     plt.tight_layout()
#     plt.savefig(f"violin_{feat}_raw_vs_corrected.png", dpi=300)
#     plt.show()
#
