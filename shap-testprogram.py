# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:17:33 2023

@author: lhh
"""


import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import KFold, GridSearchCV,cross_val_score,train_test_split
from sklearn.metrics import precision_recall_curve,auc,roc_curve,accuracy_score
import matplotlib.pyplot as plt
import shap
import matplotlib

seed=7
np.random.seed(seed)

#######调用数据
train_data=pd.read_csv('diabetes-normal-train.tsv',header='infer',sep="\t")
test_data=pd.read_csv('diabetes-normal-test.tsv',header='infer',sep="\t")

X=train_data.iloc[:,4:27]
y=train_data.iloc[:,27]
X_test=test_data.iloc[:,4:27]
y_test=test_data.iloc[:,27]
########调整参数
xgb_model=xgb.XGBClassifier(learning_rate=0.1,n_estimators=100,max_depth=3,colsample_bytree=0.8,objective='binary:logistic')

parameters={'learning_rate':[0.1,0.01,0.001],'n_estimators':[50,100,200],'max_depth':[3,6,9],'colsample_bytree':[0.4,0.6,0.8,1.0]}

clf=GridSearchCV(xgb_model,parameters,cv=10,scoring='accuracy')
clf.fit(X,y)

print("Best Params:",clf.best_params_)
print("Validation Accuracy:",clf.best_score_)

clf.best_params_
clf.best_estimator_

##########用获得的参数构建模型
xgb_model=clf.best_estimator_

xgb_model.fit(X,y)
shap.initjs()

explainer=shap.Explainer(xgb_model)
shap_values=explainer(X)
feature_names=list(X.columns)


shap.plots.waterfall(shap_values[0],max_display=20)
shap.plots.force(shap_values[0],matplotlib=True) 
shap.plots.beeswarm(shap_values,max_display=20)
fig=shap.plots.bar(shap_values,max_display=20)
plt.show()
plt.savefig('1.png')

plt.rcParams['font.sans-serif']=['SimHei']#特证名可显示中文
plt.rcParams['axes.unicode_minus']=False
shap.plots.waterfall(shap_values[0],max_display=20,show=False)
plt.savefig('1.pdf',bbox_inches='tight')
plt.close()
help(plt.rcParams)


plt.rcParams['font.sans-serif']=['Times New Roman']#特证名可显示中文
plt.rcParams['axes.unicode_minus']=False
shap.plots.waterfall(shap_values[0],max_display=20,show=False)
plt.savefig('2.pdf',bbox_inches='tight')
plt.close()
plt.rcParams.keys()
plt.rcParams.update(plt.rcParamsDefault)

shap.plots.beeswarm(shap_values,max_display=20,show=False)
plt.savefig('beeswarm.pdf')
plt.close()

shap.plots.force(shap_values,matplotlib=True)
feature_names1=['WBC','MO%','MO#','RBC','RDW-CV','RDW-SD','MCV','HCT','LY%','LY#','BASO%','BASO#','EOS%','EOS#','HGB','MCH','MCHC','PLT','MPV','PDW','PCT','NE%','NE#']
X.columns=feature_names1
shap.s
