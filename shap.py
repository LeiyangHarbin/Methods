# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 19:39:25 2023

@author: Li Honghao
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
train_data=pd.read_csv('diabetes-normal-train.tsv',header='infer',sep="\t")
test_data=pd.read_csv('diabetes-normal-test.tsv',header='infer',sep="\t")

feature_names1=['WBC','MO%','MO','RBC','RDW-CV','RDW-SD','MCV','HCT','LY%','LY','BASO%','BASO','EOS%','EOS','HGB','MCH','MCHC','PLT','MPV','PDW','PCT','NE%','NE']
X=train_data.iloc[:,4:27]
y=train_data.iloc[:,27]
X_test=test_data.iloc[:,4:27]
y_test=test_data.iloc[:,27]
X.columns=feature_names1
X_test.columns=feature_names1
########调整参数
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

##########用获得的参数构建模型
xgb_model=xgb.XGBClassifier(learning_rate=learning_rate,n_estimators=n_estimators,max_depth=max_depth,colsample_bytree=colsample_bytree,objective='binary:logistic')

xgb_model.fit(X,y)

###############shap
import shap#在开始的时候运行可以导入，如果先训练模型导入shap的时候会报错找不到initjs
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

