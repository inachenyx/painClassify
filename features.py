from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import scikitplot as skplt
from sklearn.tree import plot_tree
"""数据集准备"""
data1= loadmat('allfeature.mat')
feature=['max','min','mean','med','peak','arv','var','std','kurtosis',\
                'skewness','rms','rs','rmsa','waveformF','peakF','impulseF','clearanceF',\
                'FC','MSF','RMSF','VF','RVF',\
                'SKMean','SKStd','SKSkewness','SKKurtosis','psdE','svdpE','eE','ApEn', 'SpEn','FuzzyEn','PeEn','enveEn',\
                'detaDE','thetaDE','alphaDE','betaDE', 'gammaDE']
finallydata=np.array(data1['allfeature'])#960*78
#print(finallydata.shape)
featuremap= {}
for i in range(0, 78):
    if i<=38:
        featuremap[i]=feature[i]+'_ch1'
    else:
        featuremap[i] = feature[i-39] + '_ch2'
#print(featuremap)
df_allfeature=pd.DataFrame({})
for i in range(0,78):
    df_allfeature[featuremap[i]] = pd.Series(finallydata[:, i])
y={}
#
featurename=list(df_allfeature)
for i in range(0,20):
    for index in range(48):
        if i*48+index<7*48:
            y[i*48+index]='mild'
        elif i*48+index<14*48:
            y[i*48+index]='middle'
        else:
            y[i * 48 + index] ='severe'
df_allfeature['class'] = pd.Series(y)
print(df_allfeature)
#print((df_allfeature.values[:,0:-1]).shape)#960*79(包含一列类别)

X, y = df_allfeature.values[:,0:-1], df_allfeature.values[:, -1]
print(y)
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print(y)
"绘制学习曲线"
# skplt.estimators.plot_learning_curve(LogisticRegression(), X, y,
#                                      cv=7, shuffle=True, scoring="accuracy",
#                                      n_jobs=-1, figsize=(6,4), title_fontsize="large", text_fontsize="large",
#                                      title="Digits Classification Learning Curve")
# plt.show()
"""pca降维"""
# from sklearn.decomposition import PCA
# pca=PCA()
# pca.fit( X )
# skplt.decomposition.plot_pca_2d_projection(pca,X,y,figsize=(8,8))
# plt.show()
# print(featurename[0:-1])
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
"""特征重要性排序"""
# RF = RandomForestClassifier(n_estimators=5,
#                                 random_state=0,
#                             n_jobs=-1)
# RF.fit(X,y)
# skplt.estimators.plot_feature_importances(RF, feature_names=featurename, x_tick_rotation=90, figsize=(8, 8))
# plt.show()
#
# print(featurename)
"""选择特征"""
selectFeature=['alphaDE_ch2','gammaDE_ch1','betaDE_ch1','PeEn_ch1','betaDE_ch2'
,'peak_ch1','alphaDE_ch1','gammaDE_ch2','PeEn_ch2','thetaDE_ch2'
,'rmsa_ch1','rs_ch2','ApEn_ch1','rmsa_ch2','RMSF_ch1','detaDE_ch1'
,'SpEn_ch1','psdE_ch2','arv_ch1','peak_ch2']
forviolin_tick=['alphaDE_ch1','gammaDE_ch1','betaDE_ch1','PeEn_ch1','alphaDE_ch2','betaDE_ch2','gammaDE_ch2','PeEn_ch2']
selectFeatureIndex=np.empty((20),dtype=int)
# print(selectFeature)

for (i,v) in enumerate(selectFeature):
    selectFeatureIndex[i]=featurename.index(v)
slectFeatureIndex= np.squeeze(selectFeatureIndex)
print(selectFeatureIndex)
print(df_allfeature.values[:,selectFeatureIndex].shape)#960*20
# X0= df_allfeature.values[:,0:-1]
X, y = df_allfeature.values[:,selectFeatureIndex], df_allfeature.values[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(y)

from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
from sklearn.datasets import make_blobs

# meta-estimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# 决策树可视化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练决策树模型
decision_tree = DecisionTreeClassifier( min_samples_split=2, min_samples_leaf=2,max_depth=4)
decision_tree.fit(X_train, y_train)

# 可视化决策树
plt.figure(figsize=(15, 10))
plot_tree(decision_tree, filled=True, feature_names=selectFeature, class_names=["mild", "middle","severe"])
plt.rcParams['font.size'] = 15  # 设置字体大小
plt.rcParams['text.color'] = 'blue'  # 设置文本颜色
plt.rcParams['axes.labelcolor'] = 'red'  # 设置坐标轴标签颜色
plt.show()

classifiers = {
    'KN': KNeighborsClassifier(n_neighbors=3),
    'SVC': SVC(kernel="sigmoid", C=0.8),
    'SVC': SVC(gamma=2, C=1),
    'DT': DecisionTreeClassifier(max_depth=5),
    'RF': RandomForestClassifier(n_estimators=10, max_depth=5, max_features=1),  # clf.feature_importances_
    'ET': ExtraTreesClassifier(n_estimators=10, max_depth=None),  # clf.feature_importances_
    'AB': AdaBoostClassifier(n_estimators=100),
    'GB': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
    'GNB': GaussianNB(),
    'LD': LinearDiscriminantAnalysis(),
    'QD': QuadraticDiscriminantAnalysis()}


# for name, clf in classifiers.items():
#     scores = cross_val_score(clf, X, y,cv=10, scoring='accuracy')
#     print(name, '\t--> ', scores.mean())
# skplt.estimators.plot_learning_curve(ExtraTreesClassifier(n_estimators=10, max_depth=None), X, y,
#                                      cv=10, shuffle=True, scoring="accuracy",
#                                      n_jobs=-1, figsize=(6,4), title_fontsize="large", text_fontsize="large",
#                                      title="ExtraTreesClassifier Classification Learning Curve")
# plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#参数优化
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
# LD
#构建LDA模型
# lda = LinearDiscriminantAnalysis()
#
# #设置超参数搜索范围
# param_distributionsLD = {
#     'solver': ['svd', 'lsqr', 'eigen'],
#     'n_components': np.arange(2, 10),
#     'tol': np.linspace(0.0001, 0.001, 10)
# }
#
# #使用RandomizedSearchCV调参
# rscv = RandomizedSearchCV(lda, param_distributionsLD, cv=10, scoring='accuracy', n_iter=20)
# rscv.fit(X, y)
#
# # 查看最佳分数
# print("LinearDiscriminantAnalysis Best score: {}".format(rscv.best_score_))
# # 查看最佳参数
# print("LinearDiscriminantAnalysis Optimal params: {}".format(rscv.best_params_))
# #DecisionTreeClassifier
# # 设定要调参的参数
# param_dist = {"max_depth": [3, 15,100,None],
#               "max_features": randint(1, 11),
#               "min_samples_split": randint(2, 11),
#               "min_samples_leaf": randint(4, 11),
#               "criterion": ["gini", "entropy"]}
#
# tree = DecisionTreeClassifier()
# # 随机搜索，搜索20次，记录最佳分数
# tree_cv = RandomizedSearchCV(tree, param_dist, cv=10, scoring='accuracy', n_iter=20)
# tree_cv.fit(X, y)
#
# # 查看最佳分数
# print("DecisionTreeClassifier Best score: {}".format(tree_cv.best_score_))
# # 查看最佳参数
# print("DecisionTreeClassifier Optimal params: {}".format(tree_cv.best_params_))
# # ET
# #定义需要搜索的参数组合
# ET = ExtraTreesClassifier()
# param_distET = {'n_estimators':[10,50,100,200],
#               'max_depth': [5, 10, 15, 20],
#               'min_samples_split': [2, 5, 10],
#               'min_samples_leaf': [1, 2, 4],
#               'bootstrap': [True, False]}
#
# #使用RandomizedSearchCV搜索参数
# random_searchET = RandomizedSearchCV(ET, param_distributions = param_distET,
#                                      cv=10, scoring='accuracy', n_iter=20)
#
# random_searchET.fit(X, y)
#
#
# # 查看最佳分数
# print("ExtraTreesClassifier Best score: {}".format(random_searchET.best_score_))
# # 查看最佳参数
# print("ExtraTreesClassifier Optimal params: {}".format(random_searchET.best_params_))
# # GB
# # 先定义参数搜索范围
# param_distGB = {'n_estimators': range(30,150),
#               'learning_rate': np.arange(0.01, 0.5, 0.01),
#               'max_depth': range(2,7),
#               'max_features': range(2,9)}
#
# # 实例化GradientBoostingClassifier
# GB = GradientBoostingClassifier()
#
# # 调用RandomizedSearchCV
# rscvGB = RandomizedSearchCV(GB, param_distributions=param_distGB,cv=10, scoring='accuracy', n_iter=20)
#
# # 进行参数搜索
# rscvGB .fit(X, y)
#
#
# # 查看最佳分数
# print("GradientBoostingClassifier Best score: {}".format(rscvGB.best_score_))
# # 查看最佳参数
# print("GradientBoostingClassifier Optimal params: {}".format(rscvGB.best_params_))

# ET = ExtraTreesClassifier(n_estimators=10, max_depth=None)
# ET.fit(X_train, y_train)
# predicted_probas = ET.predict_proba(X_test)
# The magic happens here
# import matplotlib.pyplot as plt
# import scikitplot as skplt
# skplt.metrics.plot_roc(y_test, predicted_probas)
# plt.show()


# predictions = cross_val_predict(ET , X, y,cv=10)
# plot = skplt.metrics.plot_confusion_matrix(y, predictions, normalize=True,cmap="Reds")
# plt.show()
# 正确率图
# DT_probas0 =cross_val_score(DecisionTreeClassifier(max_depth=5), X0, y,cv=10, scoring='accuracy')
# DT_mean0=DT_probas0.mean()
# DT_errors0=DT_probas0.std()

# DT_acc =cross_val_score(DecisionTreeClassifier(max_depth=5), X, y,cv=10, scoring='accuracy')
# DTacc_mean=DT_acc.mean()
# DTacc_errors=DT_acc.std()
#
# DT_f1 =cross_val_score(DecisionTreeClassifier(max_depth=5), X, y,cv=10, scoring='f1_macro')
# DTf1_mean=DT_f1.mean()
# DTf1_errors=DT_f1.std()
#
# DT_pre=cross_val_score(DecisionTreeClassifier(max_depth=5), X, y,cv=10, scoring='precision_macro')
# DTpre_mean=DT_pre.mean()
# DTpre_errors=DT_pre.std()
#
# DT_recall =cross_val_score(DecisionTreeClassifier(max_depth=5), X, y,cv=10, scoring='recall_macro')
# DTrecall_mean=DT_recall.mean()
# DTrecall_errors=DT_recall.std()
#
# # ET_probas0 = cross_val_score(ExtraTreesClassifier(n_estimators=10, max_depth=None), X0, y,cv=10, scoring='accuracy')
# # ET_mean0=ET_probas0.mean()
# # ET_errors0=ET_probas0.std()
#
#
#
# ET_acc =cross_val_score(ExtraTreesClassifier(n_estimators=10, max_depth=None), X, y,cv=10, scoring='accuracy')
# ETacc_mean=ET_acc.mean()
# ETacc_errors=ET_acc.std()
#
# ET_f1 =cross_val_score(ExtraTreesClassifier(n_estimators=10, max_depth=None), X, y,cv=10, scoring='f1_macro')
# ETf1_mean=ET_f1.mean()
# ETf1_errors=ET_f1.std()
#
# ET_pre=cross_val_score(ExtraTreesClassifier(n_estimators=10, max_depth=None), X, y,cv=10, scoring='precision_macro')
# ETpre_mean=ET_pre.mean()
# ETpre_errors=ET_pre.std()
#
# ET_recall =cross_val_score(ExtraTreesClassifier(n_estimators=10, max_depth=None), X, y,cv=10, scoring='recall_macro')
# ETrecall_mean=ET_recall.mean()
# ETrecall_errors=ET_recall.std()
#
# # GB_probas0 =cross_val_score(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0), X0, y,cv=10, scoring='accuracy')
# # GB_mean0=GB_probas0.mean()
# # GB_errors0=GB_probas0.std()
#
#
# GB_acc =cross_val_score(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0), X, y,cv=10, scoring='accuracy')
# GBacc_mean=GB_acc.mean()
# GBacc_errors=GB_acc.std()
#
# GB_f1 =cross_val_score(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0), X, y,cv=10, scoring='f1_macro')
# GBf1_mean=GB_f1.mean()
# GBf1_errors=GB_f1.std()
#
# GB_pre=cross_val_score(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0), X, y,cv=10, scoring='precision_macro')
# GBpre_mean=GB_pre.mean()
# GBpre_errors=GB_pre.std()
#
# GB_recall =cross_val_score(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0), X, y,cv=10, scoring='recall_macro')
# GBrecall_mean=GB_recall.mean()
# GBrecall_errors=GB_recall.std()
#
# # LD_probas0 = cross_val_score(LinearDiscriminantAnalysis(), X0, y,cv=10, scoring='accuracy')
# # LD_mean0=LD_probas0.mean()
# # LD_errors0=LD_probas0.std()
#
#
#
# LD_acc =cross_val_score(LinearDiscriminantAnalysis(), X, y,cv=10, scoring='accuracy')
# LDacc_mean=LD_acc.mean()
# LDacc_errors=LD_acc.std()
#
# LD_f1 =cross_val_score(LinearDiscriminantAnalysis(), X, y,cv=10, scoring='f1_macro')
# LDf1_mean=LD_f1.mean()
# LDf1_errors=LD_f1.std()
#
# LD_pre=cross_val_score(LinearDiscriminantAnalysis(), X, y,cv=10, scoring='precision_macro')
# LDpre_mean=LD_pre.mean()
# LDpre_errors=LD_pre.std()
#
# LD_recall =cross_val_score(LinearDiscriminantAnalysis(), X, y,cv=10, scoring='recall_macro')
# LDrecall_mean=LD_recall.mean()
# LDrecall_errors=LD_recall.std()
#
# # vote_probas0=cross_val_score(VotingClassifier(
# #      estimators=[('DT',classifiers['DT']), ('ET', classifiers['ET']), ('GB', classifiers['GB']),('LD',classifiers['LD'])],
# #      voting='soft',
# #      weights=[2,20,5,1]), X0, y,cv=10, scoring='accuracy')
# # vote_mean0=vote_probas0.mean()
# # vote_errors0=vote_probas0.std()
#
# # vote_probas=cross_val_score(VotingClassifier(
# #      estimators=[('DT',classifiers['DT']), ('ET', classifiers['ET']), ('GB', classifiers['GB']),('LD',classifiers['LD'])],
# #      voting='soft',
# #      weights=[2,20,5,1]), X, y,cv=10, scoring='accuracy')
# # vote_mean=vote_probas.mean()
# # vote_errors=vote_probas.std()
# vote_acc =cross_val_score(VotingClassifier(
#      estimators=[('DT',classifiers['DT']), ('ET', classifiers['ET']), ('GB', classifiers['GB']),('LD',classifiers['LD'])],
#      voting='soft',
#      weights=[2,15,5,1]), X, y,cv=10, scoring='accuracy')
# voteacc_mean=vote_acc.mean()
# voteacc_errors=vote_acc.std()
#
# vote_f1 =cross_val_score(VotingClassifier(
#      estimators=[('DT',classifiers['DT']), ('ET', classifiers['ET']), ('GB', classifiers['GB']),('LD',classifiers['LD'])],
#      voting='soft',
#      weights=[2,15,5,1]), X, y,cv=10, scoring='f1_macro')
# votef1_mean=vote_f1.mean()
# votef1_errors=vote_f1.std()
#
# vote_pre=cross_val_score(VotingClassifier(
#      estimators=[('DT',classifiers['DT']), ('ET', classifiers['ET']), ('GB', classifiers['GB']),('LD',classifiers['LD'])],
#      voting='soft',
#      weights=[2,15,5,1]), X, y,cv=10, scoring='precision_macro')
# votepre_mean=vote_pre.mean()
# votepre_errors=vote_pre.std()
#
# vote_recall =cross_val_score(VotingClassifier(
#      estimators=[('DT',classifiers['DT']), ('ET', classifiers['ET']), ('GB', classifiers['GB']),('LD',classifiers['LD'])],
#      voting='soft',
#      weights=[2,15,5,1]), X, y,cv=10, scoring='recall_macro')
# voterecall_mean=vote_recall.mean()
# voterecall_errors=vote_recall.std()
#
# clf_names=['LD','DT',
#            'ET',
#            'GB',
#            'VOTE']


# scores=np.array([knn_classifier.score(X_test, y_test),svc_classifier.score(X_test, y_test),
#                        dt_classifier.score(X_test, y_test)])
#
# scores0=np.array([DT_mean0,ET_mean0,GB_mean0,LD_mean0,vote_mean0])
# errors0=np.array([DT_errors0,ET_errors0,GB_errors0,LD_errors0,vote_errors0])
#
#
# scores=np.array([DT_mean,ET_mean,GB_mean,LD_mean,vote_mean])
# errors=np.array([DT_errors,ET_errors,GB_errors,LD_errors,vote_errors])

# f1=np.array([LDf1_mean,DTf1_mean,ETf1_mean,GBf1_mean,votef1_mean])
# f1errors=np.array([LDf1_errors,DTf1_errors,ETf1_errors,GBf1_errors,votef1_errors])
#
# pre=np.array([LDpre_mean,DTpre_mean,ETpre_mean,GBpre_mean,votepre_mean])
# preerrors=np.array([LDpre_errors,DTpre_errors,ETpre_errors,GBpre_errors,votepre_errors])
#
# recall=np.array([LDrecall_mean,DTrecall_mean,ETrecall_mean,GBrecall_mean,voterecall_mean])
# recallerrors=np.array([LDrecall_errors,DTrecall_errors,ETrecall_errors,GBrecall_errors,voterecall_errors])
#
# acc=np.array([LDacc_mean,DTacc_mean,ETacc_mean,GBacc_mean,voteacc_mean])
# accerrors=np.array([LDacc_errors,DTacc_errors,ETacc_errors,GBacc_errors,voteacc_errors])
#
#
# fenleiqi=np.array(['LD',"DT", "ET", "GB",'VOTE'])
# x_len = np.arange(len(fenleiqi))
# print(x_len)
# total_width, n = 0.9, 4
# width = 0.2
# xticks = x_len - (total_width - width) / 2
# #plt.bar(fenleiqi,scores,width=0.3,yerr=errors,error_kw = {'ecolor' : '0.2', 'capsize' :6 }, alpha=0.7)
# plt.rcParams['ytick.direction'] = 'in'
# plt.rcParams['xtick.direction'] = 'in'
# font = {'family': 'Arial'  # 'serif',
#         #         ,'style':'italic'
#     , 'weight': 'bold'  # 'normal'
#         #         ,'color':'red'
#     , 'size': 16
#         }
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), dpi=100, facecolor="w")
# plt.bar(xticks, f1, width=0.9*width,color="#6788ED",label="OriginalFeature",yerr = f1errors,error_kw = {'ecolor' : '0.2', 'capsize' :6 }, alpha=0.8)
# plt.bar(xticks+width, pre, width=0.9*width,color="#99BAFE",label="SelectFeature",yerr = preerrors,error_kw = {'ecolor' : '0.2', 'capsize' :6 }, alpha=0.8)
# plt.bar(xticks+2*width, recall, width=0.9*width,color="#F6A789",label="SelectFeature",yerr =recallerrors,error_kw = {'ecolor' : '0.2', 'capsize' :6 }, alpha=0.8)
# plt.bar(xticks+3*width, acc, width=0.9*width,color="#E16852",label="SelectFeature",yerr = accerrors,error_kw = {'ecolor' : '0.2', 'capsize' :6 }, alpha=0.8)
# plt.xticks(x_len,fenleiqi, fontproperties=font)
#
# # for m, n ,z in zip(fenleiqi, scores,errors):
# #     # ha: horizontal alignment
# #     # va: vertical alignment
# #     plt.text(m, n +z, '%.2f' %n, ha='center', va='bottom')
# # num2=1表示legend位于图像上侧水平线(其它设置：num1=1.05,num3=3,num4=0)。
# num1 = 2
# num2 = 1.2
# num3 = 3
# num4 = 0
# plt.legend(prop={'family' : 'Arial', 'size': 14,'weight': 'bold'}, labels=['F1','Precision','Recall','Accuracy'],ncol=4,bbox_to_anchor=(0,1.01,1.0,2), loc=8)
#
# for m, n ,z in zip(xticks, f1,f1errors):
#     # ha: horizontal alignment
#     # va: vertical alignment
#     plt.text(m, n +z, '%.2f' %n, ha='center', va='bottom',fontdict = font)
# for m, n ,z in zip(xticks, pre,preerrors):
#     # ha: horizontal alignment
#     # va: vertical alignment
#     plt.text(m+width, n +z, '%.2f' %n, ha='center', va='bottom',fontdict = font)
# for m, n ,z in zip(xticks, recall,recallerrors):
#     # ha: horizontal alignment
#     # va: vertical alignment
#     plt.text(m+2*width, n +z, '%.2f' %n, ha='center', va='bottom',fontdict = font)
# for m, n ,z in zip(xticks, acc,accerrors):
#     # ha: horizontal alignment
#     # va: vertical alignment
#     plt.text(m+3*width, n +z, '%.2f' %n, ha='center', va='bottom',fontdict = font)
#
# plt.xlabel("Classifiers",fontdict = font)
# plt.ylabel("Scores",fontdict = font)
# # plt.title('Accuracy Rate',fontdict = font)
# plt.ylim(0, 1)
# ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
# ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontproperties=font)
# ax.spines['right'].set_linewidth(1.5)
# ax.spines['bottom'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['top'].set_linewidth(1.5)
# plt.show()
# #混淆矩阵
# predictions = cross_val_predict(VotingClassifier(
#      estimators=[('DT',classifiers['DT']), ('ET', classifiers['ET']), ('GB', classifiers['GB']),('LD',classifiers['LD'])],
#      voting='soft',
#      weights=[2,15,5,1]) , X, y,cv=10)
# plot = skplt.metrics.plot_confusion_matrix(y, predictions, normalize=True,cmap='Pastel1_r')
# plt.show()
# 学习曲线
# skplt.estimators.plot_learning_curve(VotingClassifier(estimators=[('DT',classifiers['DT']), ('ET', classifiers['ET']), ('GB', classifiers['GB']),('LD',classifiers['LD'])],voting='soft',weights=[2,15,5,1]), X, y,
#                                      cv=10, shuffle=True, scoring="accuracy",
#                                      n_jobs=-1, figsize=(6,4), title_fontsize="large", text_fontsize="large",
#                                      title="VoteClassifier Learning Curve")
# plt.show()
# 特征重要性
# RF = RandomForestClassifier(n_estimators=5,
#                                 random_state=0,
#                             n_jobs=-1)
# # RF.fit(X0,y)
# skplt.estimators.plot_feature_importances(RF, feature_names=featurename, x_tick_rotation=90, figsize=(8, 8))
#
# print(featurename)
# plt.show()
# ROC曲线

# LD=LinearDiscriminantAnalysis()
#
# LD.fit(X_train, y_train)
# predicted_probasLD = LD.predict_proba(X_test)
# #
# DT=DecisionTreeClassifier(max_depth=5)
#
# DT.fit(X_train, y_train)
# predicted_probasDT = DT.predict_proba(X_test)
# #
# ET=ExtraTreesClassifier(n_estimators=10, max_depth=None)
#
# ET.fit(X_train, y_train)
# predicted_probasET = ET.predict_proba(X_test)
# #
# GB=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
#
# GB.fit(X_train, y_train)
# predicted_probasGB = GB.predict_proba(X_test)
# #
# VT=VotingClassifier(
#      estimators=[('DT',classifiers['DT']), ('ET', classifiers['ET']), ('GB', classifiers['GB']),('LD',classifiers['LD'])],
#      voting='soft',
#      weights=[2,15,5,1])
#
# VT.fit(X_train, y_train)
# predicted_probasVT = VT.predict_proba(X_test)
# #The magic happens here
# import matplotlib.pyplot as plt
# import scikitplot as skplt
#skplt.metrics.plot_roc(y_test, 1,predicted_probasLD,predicted_probasDT,predicted_probasET,predicted_probasGB,predicted_probasVT,cmap=plt.cm.OrRd)
# plt.show()

#绘制有显著性检验结果的小提琴图
# import seaborn as sns
# from statannotations.Annotator import Annotator
# def forviolin(ax,df,feature):
#    # fig, ax= plt.subplots(figsize=(5, 4), dpi=100, facecolor="w")
#     ax = sns.violinplot(x=df["class"], y=df[feature],palette=['#99BAFE','#F6A789','#E16852'],ax=ax)
#     pairs = [("mild", "middle"), ("middle", "severe"), ("mild", "severe")]
#     annotator = Annotator(ax, pairs, x=df["class"], y=df[feature], )
#     annotator.configure(test='Mann-Whitney', text_format='star', line_height=0.03, line_width=1)
#     annotator.apply_and_annotate()
#
#     ax.tick_params(which='major', direction='in', length=3, width=1., labelsize=14, bottom=False)
#     for spine in ["top", "left", "right"]:
#         ax.spines[spine].set_visible(False)
#     ax.spines['bottom'].set_linewidth(1.5)
#     ax.grid(axis='y', ls='--', c='gray')
#     ax.set_axisbelow(True)
#     font = {'family': 'Arial'  # 'serif',
#         #         ,'style':'italic'
#     , 'weight': 'bold'  # 'normal'
#         #         ,'color':'red'
#     , 'size': 16
#         }
#     ax.set_xlabel('class', fontdict = font)
#     ax.set_ylabel(feature, fontdict = font)
#     return ax
#
# fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(5, 4), dpi=100, facecolor="w")
# for j in range(0, 2):
#     for i in range(0, 4):
#         forviolin(ax[j, i], df_allfeature, forviolin_tick[4*j+i])
#
# # fig1, ax1 = plt.subplots(nrows=2, ncols=5, figsize=(5, 4), dpi=100, facecolor="w")
# # for j in range(0, 2):
# #     for i in range(0, 5):
# #         forviolin(ax1[j, i], df_allfeature, selectFeature[5*j+10+i])
# plt.show()