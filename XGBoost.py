# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:50:43 2023

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
train_data=pd.read_csv('diabetes-normal-train.tsv',header='infer',sep="\t")
test_data=pd.read_csv('diabetes-normal-test.tsv',header='infer',sep="\t")

X=train_data.iloc[:,4:27]
y=train_data.iloc[:,27]
X_test=test_data.iloc[:,4:27]
y_test=test_data.iloc[:,27]
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

######训练集预测
y_train_pre=xgb_model.predict(X)
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
index.to_csv('index-XGBoost.tsv',sep='\t')

########
import seaborn as sns

cm=confusion_matrix(y,y_train_pre)
cm1=pd.DataFrame(cm,index=['normal','diabetes'],columns=['normal','diabetes'])

plt.figure(figsize=(8, 8))
sns.heatmap(cm1, annot=True,fmt='.20g',cmap='Blues')#添加fmt，不使用科学计数法显示

plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig('confusion matrix-train.pdf')
plt.show()

##############绘制ROC曲线
y_train_prob=xgb_model.predict_proba(X)
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

axes[0].savefig('diabetes-normal-train-ROC.pdf')

##########测试集预测
y_test_pre=xgb_model.predict(X_test)
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
index.to_csv('index-XGBoost-test.tsv',sep='\t')

########混淆矩阵
cm=confusion_matrix(y_test,y_test_pre)
cm1=pd.DataFrame(cm,index=['normal','diabetes'],columns=['normal','diabetes'])

plt.figure(figsize=(8, 8))
sns.heatmap(cm1, annot=True,fmt='.20g',cmap='Blues')#添加fmt，不使用科学计数法显示

plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig('confusion matrix-test.pdf')
plt.show()
######test-ROC
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

axes[0].savefig('diabetes-normal-test-ROC.pdf')

#############validation
validation=pd.read_csv('diabetes-validation.tsv',header='infer',sep="\t")
X_val=validation.iloc[:,0:23]
y_val=validation.iloc[:,23]

y_val_pred=xgb_model.predict(X_val)
acc=accuracy_score(y_val,y_val_pred)

f=open("diabetes-validation accuracy score.txt","w")
f.write(str(acc))
f.close()

#########XGBoost特征重要性
feature=pd.read_csv('血常规指标中英文对照.txt',delimiter='\t',header=None)
feature_names=feature.iloc[:,1]
feature_importances=xgb_model.feature_importances_
indices = np.argsort(feature_importances)[::-1]
for index in indices:
    print("特征 %s 重要度为 %f" %(feature_names[index], feature_importances[index]))

importances=pd.DataFrame(feature_importances)
importances.index=feature_names
importances.columns=['feature_importances']
importances_order=importances.sort_values(by=['feature_importances'],ascending=False)
importances_order.to_csv('feature importances.tsv',sep='\t')

plt.figure(figsize=(16,8))
plt.title("Feature importances",fontsize=30)
plt.bar(range(len(feature_importances)), feature_importances[indices], color='deepskyblue')
plt.xticks(range(len(feature_importances)), np.array(feature_names)[indices], rotation=90,color='k',fontsize=20)
plt.savefig('feature importances.pdf',bbox_inches='tight')

plt.figure(figsize=(16,8))
plt.title("Feature Importances-10",fontsize=30,fontname='Times New Roman')
plt.bar(range(len(feature_importances))[0:10], feature_importances[indices][0:10], color='#F78179')
plt.xticks(range(len(feature_importances))[0:10], np.array(feature_names)[indices][0:10], rotation=90,color='black',fontsize=20,fontname='Times New Roman')
plt.yticks(fontsize=20,fontname='Times New Roman')
plt.savefig('feature importances-10.pdf',bbox_inches='tight')
