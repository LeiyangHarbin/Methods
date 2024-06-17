# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:52:47 2023

@author: lhh
"""



import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import KFold, GridSearchCV,cross_val_score,train_test_split
from sklearn.metrics import precision_recall_curve,auc,roc_curve,accuracy_score,confusion_matrix
import matplotlib.pyplot as plt

seed=7
np.random.seed(seed)

#######调用数据
train=pd.read_csv('normal-diabetes-II-only-train.tsv',header='infer',sep="\t")
test=pd.read_csv('normal-diabetes-II-only-test.tsv',header='infer',sep="\t")

X=train.iloc[:,0:23]
y=train.iloc[:,23]
X_test=test.iloc[:,0:23]
y_test=test.iloc[:,23]

#######调整参数
xgb_model=xgb.XGBClassifier(learning_rate=0.1,n_estimators=100,max_depth=3,colsample_bytree=0.8,objective='binary:logistic')

parameters={'learning_rate':[0.1,0.01,0.001],'n_estimators':[50,100,200],'max_depth':[3,6,9],'colsample_bytree':[0.6,0.8,1.0]}

clf=GridSearchCV(xgb_model,parameters,cv=10,scoring='accuracy')
clf.fit(X,y)

print("Best Params:",clf.best_params_)
print("Validation Accuracy:",clf.best_score_)
clf.best_estimator_

##########用获得的参数构建模型
xgb_model=clf.best_estimator_

kf = KFold(n_splits=10, shuffle=True, random_state=seed)

results=cross_val_score(xgb_model, X,y,cv=kf)
print('Standardize: %.3f (%.3f) MSE' % (results.mean(),results.std()))

print(results.mean())
result1=pd.DataFrame(results)
result_mean=results.mean()
result1.loc[len(result1.index)] = result_mean
result1.index=[1,2,3,4,5,6,7,8,9,10,'mean']
result1.columns=['cross_val_score']
result1.to_csv('cross_val_score.tsv',sep='\t')

xgb_model.fit(X,y)

###########绘制ROC曲线
y_prob=xgb_model.predict_proba(X)
fpr,tpr,thresholds=roc_curve(y,y_prob[:,1])
roc_auc=auc(fpr,tpr)

axes=plt.subplots(1,1,figsize=(6,6),dpi=300)
lab='Overall.AUC=%.4f' % (roc_auc)

axes[1].step(fpr,tpr,label=lab,lw=2,color='red')
axes[1].set_title('ROC curve',fontsize=12,fontname='Times New Roman')
axes[1].set_xlabel('False Positive Rate',fontsize=12,fontname='Times New Roman')
axes[1].set_ylabel('True Positive Rate',fontsize=12,fontname='Times New Roman')
axes[1].legend(loc='lower right',prop={'family':'Times New Roman','size':12})

plt.show()

axes[0].savefig('normal-diabetes-II-only-train-ROC.pdf')

############绘制test-ROC曲线
y_test_prob=xgb_model.predict_proba(X_test)
fpr,tpr,thresholds=roc_curve(y_test,y_test_prob[:,1])
roc_auc=auc(fpr,tpr)

axes=plt.subplots(1,1,figsize=(6,6),dpi=300)
lab='Overall.AUC=%.4f' % (roc_auc)

axes[1].step(fpr,tpr,label=lab,lw=2,color='red')
axes[1].set_title('ROC curve',fontsize=12,fontname='Times New Roman')
axes[1].set_xlabel('False Positive Rate',fontsize=12,fontname='Times New Roman')
axes[1].set_ylabel('True Positive Rate',fontsize=12,fontname='Times New Roman')
axes[1].legend(loc='lower right',prop={'family':'Times New Roman','size':12})

plt.show()

axes[0].savefig('normal-diabetes-II-only-test-ROC.pdf')

###############shap
import shap
shap.initjs()
explainer=shap.Explainer(xgb_model)
shap_values=explainer(X)

shap.plots.force(shap_values)
shap.save_html('shap-feature-all.html',shap.plots.force(shap_values))

shap.plots.force(shap_values[0],matplotlib=True,show=False)#这是两个不同分类的特征贡献的两个例子
plt.savefig('shap-feature-single.pdf',bbox_inches='tight')
shap.save_html('shap-feature-single.html',shap.plots.force(shap_values[0]))
shap.plots.force(shap_values[2200],matplotlib=True,show=False)
plt.savefig('shap-feature-single-1.pdf',bbox_inches='tight')
shap.save_html('shap-feature-single-1.html',shap.plots.force(shap_values[2200]))

shap.plots.beeswarm(shap_values,max_display=23)#这个图片目前没办法保存
shap.plots.beeswarm(shap_values,max_display=11)#这个用截图保存

shap.plots.bar(shap_values,max_display=23,show=False)#show=False可以保存图片
plt.savefig('shap-plot-bar.pdf',bbox_inches='tight')

shap.plots.waterfall(shap_values[0],max_display=23,show=False)#两个不同分类的特征贡献的例子
plt.savefig('shap-plot-waterfall-single.pdf',bbox_inches='tight')
shap.plots.waterfall(shap_values[2200],max_display=23,show=False)
plt.savefig('shap-plot-waterfall-single-1.pdf',bbox_inches='tight')

#####测试集shap分析
shap_values=explainer(X_test)

shap.plots.force(shap_values)
shap.save_html('shap-feature-all-test.html',shap.plots.force(shap_values))

shap.plots.force(shap_values[0],matplotlib=True,show=False)#这是两个不同分类的特征贡献的两个例子
plt.savefig('shap-feature-single-test.pdf',bbox_inches='tight')
shap.save_html('shap-feature-single-test.html',shap.plots.force(shap_values[0]))
shap.plots.force(shap_values[1000],matplotlib=True,show=False)
plt.savefig('shap-feature-single-test-1.pdf',bbox_inches='tight')
shap.save_html('shap-feature-single-test-1.html',shap.plots.force(shap_values[1000]))

shap.plots.beeswarm(shap_values,max_display=23)#这个图片目前没办法保存
shap.plots.beeswarm(shap_values,max_display=11)#这个用截图保存

shap.plots.bar(shap_values,max_display=23,show=False)#show=False可以保存图片
plt.savefig('shap-plot-bar-test.pdf',bbox_inches='tight')

shap.plots.waterfall(shap_values[0],max_display=23,show=False)#两个不同分类的特征贡献的例子
plt.savefig('shap-plot-waterfall-single-test.pdf',bbox_inches='tight')
shap.plots.waterfall(shap_values[1000],max_display=23,show=False)
plt.savefig('shap-plot-waterfall-single-test-1.pdf',bbox_inches='tight')

##########用并发症样本的数据测试
complication=pd.read_csv('diabetes-with-complication-feature-label.tsv',header='infer',sep="\t")
X_com=complication.iloc[:,0:23]
y_com=complication.iloc[:,23]

y_com_pre=xgb_model.predict(X_com)
acc=accuracy_score(y_com_pre,y_com)

###############混淆矩阵
import seaborn as sns

y_pred=xgb_model.predict(X)
cm=confusion_matrix(y,y_pred)
cm1=pd.DataFrame(cm,index=['normal','diabetes-II'],columns=['normal','diabetes-II'])

plt.figure(figsize=(8, 8))
sns.heatmap(cm1, annot=True,fmt='.20g',cmap='Blues')#添加fmt，不使用科学计数法显示

plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig('confusion matrix-train.pdf')
plt.show()


y_test_pred=xgb_model.predict(X_test)
cm=confusion_matrix(y_test,y_test_pred)
cm1=pd.DataFrame(cm,index=['normal','diabetes-II'],columns=['normal','diabetes-II'])

plt.figure(figsize=(8, 8))
sns.heatmap(cm1, annot=True,fmt='.20g',cmap='Blues')#添加fmt，不使用科学计数法显示

plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig('confusion matrix-test.pdf')
plt.show()


