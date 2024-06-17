# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 19:53:06 2023

@author: Li Honghao
"""

import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import KFold, GridSearchCV,cross_val_score,train_test_split
from sklearn.metrics import precision_recall_curve,auc,roc_curve,accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import shap#在开始的时候运行可以导入，如果先训练模型导入shap的时候会报错找不到initjs

seed=7
np.random.seed(seed)


#######调用数据
train_data=pd.read_csv('train.tsv',header='infer',sep="\t")
test_data=pd.read_csv('test.tsv',header='infer',sep="\t")

X=train_data.iloc[:,:78]
y=train_data.iloc[:,78]
X_test=test_data.iloc[:,:78]
y_test=test_data.iloc[:,78]

######调参
xgb_model=xgb.XGBClassifier(learning_rate=0.1,n_estimators=100,max_depth=3,colsample_bytree=0.8,objective='binary:logistic')

parameters={'learning_rate':[0.1,0.01,0.001],'n_estimators':[50,100,200],'max_depth':[3,6,9],'colsample_bytree':[0.6,0.8,1.0]}

clf=GridSearchCV(xgb_model,parameters,cv=10,scoring='accuracy')
clf.fit(X,y)

print("Best Params:",clf.best_params_)
print("Validation Accuracy:",clf.best_score_)
best_params=clf.best_params_
learning_rate=best_params.get('learning_rate')
n_estimators=best_params.get('n_estimators')
max_depth=best_params.get('max_depth')
colsample_bytree=best_params.get('colsample_bytree')

#####交叉验证
xgb_model=xgb.XGBClassifier(learning_rate=learning_rate,n_estimators=n_estimators,max_depth=max_depth,colsample_bytree=colsample_bytree,objective='binary:logistic')

kf = KFold(n_splits=10, shuffle=True, random_state=seed)

results=cross_val_score(xgb_model, X,y,cv=kf)
print('Standardize: %.3f (%.3f) MSE' % (results.mean(),results.std()))

print(results.mean())
result1=pd.DataFrame(results)
result_mean=results.mean()
result1.iloc[9,0]=result_mean
result1.index=[1,2,3,4,5,6,7,8,9,'mean']
result1.columns=['cross_val_score']
result1.to_csv('cross_val_score.tsv',sep='\t')

xgb_model.fit(X,y)

###########绘制ROC曲线
y_prob=xgb_model.predict_proba(X)
fpr,tpr,thresholds=roc_curve(y,y_prob[:,1])
roc_auc=auc(fpr,tpr)

axes=plt.subplots(1,1,figsize=(6,6),dpi=300)
lab='Overall.AUC=%.8f' % (roc_auc)

axes[1].step(fpr,tpr,label=lab,lw=2,color='red')
axes[1].set_title('ROC curve',fontsize=12,fontname='Times New Roman')
axes[1].set_xlabel('False Positive Rate',fontsize=12,fontname='Times New Roman')
axes[1].set_ylabel('True Positive Rate',fontsize=12,fontname='Times New Roman')
axes[1].legend(loc='lower right',prop={'family':'Times New Roman','size':12})

plt.show()

axes[0].savefig('train-ROC.pdf')

###############shap
shap.initjs()
explainer=shap.Explainer(xgb_model)
shap_values=explainer(X)

shap.plots.force(shap_values)
shap.save_html('shap-feature-all.html',shap.plots.force(shap_values))

shap.plots.force(shap_values[0],matplotlib=True,show=False)#这是两个不同分类的特征贡献的两个例子
plt.savefig('shap-feature-single.pdf',bbox_inches='tight')
shap.save_html('shap-feature-single.html',shap.plots.force(shap_values[0]))
shap.plots.force(shap_values[1],matplotlib=True,show=False)
plt.savefig('shap-feature-single-1.pdf',bbox_inches='tight')
shap.save_html('shap-feature-single-1.html',shap.plots.force(shap_values[1]))

shap.plots.beeswarm(shap_values,max_display=78)#右键图片保存
shap.plots.beeswarm(shap_values,max_display=21)
shap.plots.beeswarm(shap_values,max_display=11)

shap.plots.bar(shap_values,max_display=78,show=False)#show=False可以保存图片
plt.savefig('shap-plot-bar.pdf',bbox_inches='tight')
shap.plots.bar(shap_values,max_display=21,show=False)#show=False可以保存图片
plt.savefig('shap-plot-bar-20.pdf',bbox_inches='tight')
shap.plots.bar(shap_values,max_display=11,show=False)#show=False可以保存图片
plt.savefig('shap-plot-bar-10.pdf',bbox_inches='tight')


shap.plots.waterfall(shap_values[0],max_display=78,show=False)#两个不同分类的特征贡献的例子
plt.savefig('shap-plot-waterfall-single.pdf',bbox_inches='tight')
shap.plots.waterfall(shap_values[1],max_display=78,show=False)
plt.savefig('shap-plot-waterfall-single-1.pdf',bbox_inches='tight')