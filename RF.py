# -*- coding: utf-8 -*-
"""
Created on Thu May  4 08:53:15 2023

@author: lhh
"""


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_recall_curve,auc,roc_curve,confusion_matrix
from sklearn.model_selection import cross_val_score,GridSearchCV,KFold
import matplotlib.pyplot as plt


seed=7
np.random.seed(seed)
###########输入数据
train_data=pd.read_csv('diabetes-normal-train.tsv',header='infer',sep="\t")
test_data=pd.read_csv('diabetes-normal-test.tsv',header='infer',sep="\t")

X=train_data.iloc[:,4:27]
y=train_data.iloc[:,27]
X_test=test_data.iloc[:,4:27]
y_test=test_data.iloc[:,27]

#########调整参数
rf=RandomForestClassifier()

params=dict(n_estimators=[20,50,100],max_depth=[10,15,20,25],
        max_leaf_nodes=[30,40,50],criterion=['gini','entropy'])

best_model=GridSearchCV(rf,param_grid=params,cv=10,scoring = 'accuracy')

best_model.fit(X,y)

best_model.best_score_
best_model.best_params_ 
best_model.best_estimator_ 

###############交叉验证
model=best_model.best_estimator_

kf = KFold(n_splits=10, shuffle=True, random_state=seed)

results=cross_val_score(model, X,y,cv=kf)
print('Standardize: %.3f (%.3f) MSE' % (results.mean(),results.std()))

print(results.mean())
result1=pd.DataFrame(results)
result_mean=results.mean()
result1.loc[len(result1.index)] = result_mean
result1.index=[1,2,3,4,5,6,7,8,9,10,'mean']
result1.columns=['cross_val_score']
result1.to_csv('cross_val_score-RF.tsv',sep='\t')

model.fit(X,y)

###########预测训练集
y_train_pre=model.predict(X)
TP=sum(a==1 and p==1 for a,p in zip (y,y_train_pre))
TN=sum(a==0 and p==0 for a,p in zip (y,y_train_pre))
FP=sum(a==0 and p==1 for a,p in zip (y,y_train_pre))
FN=sum(a==1 and p==0 for a,p in zip (y,y_train_pre))

Sn=TP/(TP+FN)
Sp=TN/(TN+FP)
Acc=(TP+TN)/(TP+TN+FP+FN)

index=pd.DataFrame()
index=index.append(pd.Series(Acc,name='Acc'))
index=index.append(pd.Series(Sn,name='Sn'))
index=index.append(pd.Series(Sp,name='Sp'))
index.to_csv('index-RF.tsv',sep='\t')

########混淆矩阵
import seaborn as sns

cm=confusion_matrix(y,y_train_pre)
cm1=pd.DataFrame(cm,index=['normal','diabetes'],columns=['normal','diabetes'])

plt.figure(figsize=(8, 8))
sns.heatmap(cm1, annot=True,fmt='.20g',cmap='Blues')#添加fmt，不使用科学计数法显示

plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig('confusion matrix-train-RF.pdf')
plt.show()
#############ROC曲线
y_train_prob=model.predict_proba(X)
fpr,tpr,thresholds=roc_curve(y,y_train_prob[:,1])
roc_auc=auc(fpr,tpr)

axes=plt.subplots(1,1,figsize=(6,6),dpi=300)
lab='Overall.AUC=%.4f' % (roc_auc)

axes[1].step(fpr,tpr,label=lab,lw=2,color='red')
axes[1].set_title('ROC curve',fontsize=12,fontname='Times New Roman')
axes[1].set_xlabel('False Positive Rate',fontsize=12,fontname='Times New Roman')
axes[1].set_ylabel('True Positive Rate',fontsize=12,fontname='Times New Roman')
axes[1].legend(loc='lower right',prop={'family':'Times New Roman','size':12})

plt.show()

axes[0].savefig('diabetes-normal-train-ROC-RF.pdf')

##########预测测试集
y_test_pre=model.predict(X_test)
TP=sum(a==1 and p==1 for a,p in zip (y_test,y_test_pre))
TN=sum(a==0 and p==0 for a,p in zip (y_test,y_test_pre))
FP=sum(a==0 and p==1 for a,p in zip (y_test,y_test_pre))
FN=sum(a==1 and p==0 for a,p in zip (y_test,y_test_pre))

Sn=TP/(TP+FN)
Sp=TN/(TN+FP)
Acc=(TP+TN)/(TP+TN+FP+FN)

index=pd.DataFrame()
index=index.append(pd.Series(Acc,name='Acc'))
index=index.append(pd.Series(Sn,name='Sn'))
index=index.append(pd.Series(Sp,name='Sp'))
index.to_csv('index-RF-test.tsv',sep='\t')

########混淆矩阵

cm=confusion_matrix(y_test,y_test_pre)
cm1=pd.DataFrame(cm,index=['normal','diabetes'],columns=['normal','diabetes'])

plt.figure(figsize=(8, 8))
sns.heatmap(cm1, annot=True,fmt='.20g',cmap='Blues')#添加fmt，不使用科学计数法显示

plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig('confusion matrix-test-RF.pdf')
plt.show()
#############ROC曲线
y_test_prob=model.predict_proba(X_test)
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

axes[0].savefig('diabetes-normal-test-ROC-RF.pdf')