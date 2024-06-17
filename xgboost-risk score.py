# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:32:13 2023

@author: Li Honghao
"""


import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import KFold, GridSearchCV,cross_val_score,train_test_split
from sklearn.metrics import precision_recall_curve,auc,roc_curve,accuracy_score
import matplotlib.pyplot as plt

seed=7
np.random.seed(seed)

#######调用数据
train_data=pd.read_csv('diabetes-normal-train.tsv',header='infer',sep="\t")
test_data=pd.read_csv('diabetes-normal-test.tsv',header='infer',sep="\t")
validation=pd.read_csv('diabetes-validation.tsv',header='infer',sep="\t")

X=train_data.iloc[:,4:27]
y=train_data.iloc[:,27]
X_test=test_data.iloc[:,4:27]
y_test=test_data.iloc[:,27]
X_val=validation.iloc[:,0:23]
y_val=validation.iloc[:,23]

########调整参数
xgb_model=xgb.XGBClassifier(learning_rate=0.1,n_estimators=100,max_depth=3,colsample_bytree=0.8,objective='binary:logistic')

parameters={'learning_rate':[0.1,0.01,0.001],'n_estimators':[50,100,200],'max_depth':[3,6,9],'colsample_bytree':[0.6,0.8,1.0]}

clf=GridSearchCV(xgb_model,parameters,cv=10,scoring='accuracy')
clf.fit(X,y)

print("Best Params:",clf.best_params_)
print("Validation Accuracy:",clf.best_score_)
clf.best_estimator_

##########用获得的参数构建模型
xgb_model=clf.best_estimator_
xgb_model.fit(X,y)

y_prob=xgb_model.predict_proba(X)
y_prob=pd.DataFrame(y_prob)
y_prob.index=X.index
y_prob.to_csv('train-risk.tsv',sep='\t')

y_test_prob=xgb_model.predict_proba(X_test)
y_test_prob=pd.DataFrame(y_test_prob)
y_test_prob.index=X_test.index
y_test_prob.to_csv('test-risk.tsv',sep='\t')

y_val_prob=xgb_model.predict_proba(X_val)
y_val_prob=pd.DataFrame(y_val_prob)
y_val_prob.index=X_val.index
y_val_prob.to_csv('validation-risk.tsv',sep='\t')
