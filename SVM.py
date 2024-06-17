# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 10:00:58 2023

@author: Li Honghao
"""


import numpy as np
import pandas as pd
from sklearn.svm import SVC
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
svm_model=SVC()

params=[
        {'kernel':['linear'],'C':[0.1,1,10,100],'max_iter':[1000000]},
        {'kernel': ['poly'], 'C': [0.1,1,10,100], 'degree': [2,3,4],'max_iter':[1000000]},
        {'kernel': ['rbf'], 'C': [0.1,1,10,100], 'gamma':[10,1,0.1,0.01]}
        ]

best_model=GridSearchCV(svm_model,param_grid=params,cv=10,scoring='accuracy')
best_model.fit(X,y)

best_model.best_score_
best_model.best_params_ 
best_model.best_estimator_ 
best_model.cv_results_

############交叉验证
model=SVC(C=0.1,kernel='linear',probability=True)##probability=true，才能运行model.predict_proba

kf = KFold(n_splits=10, shuffle=True, random_state=seed)

results=cross_val_score(model, X,y,cv=kf)
print('Standardize: %.3f (%.3f) MSE' % (results.mean(),results.std()))

print(results.mean())
result1=pd.DataFrame(results)
result_mean=results.mean()
result1.loc[len(result1.index)] = result_mean
result1.index=[1,2,3,4,5,6,7,8,9,10,'mean']
result1.columns=['cross_val_score']
result1.to_csv('cross_val_score-SVM.tsv',sep='\t')

model.fit(X,y)

#######预测训练集
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
index.to_csv('index-SVM.tsv',sep='\t')

########混淆矩阵
import seaborn as sns

cm=confusion_matrix(y,y_train_pre)
cm1=pd.DataFrame(cm,index=['normal','diabetes'],columns=['normal','diabetes'])

plt.figure(figsize=(8, 8))
sns.heatmap(cm1, annot=True,fmt='.20g',cmap='Blues')#添加fmt，不使用科学计数法显示

plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig('confusion matrix-train-SVM.pdf')
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

axes[0].savefig('diabetes-normal-train-ROC-SVM.pdf')

#######预测测试集
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
index.to_csv('index-SVM-test.tsv',sep='\t')

########混淆矩阵

cm=confusion_matrix(y_test,y_test_pre)
cm1=pd.DataFrame(cm,index=['normal','diabetes'],columns=['normal','diabetes'])

plt.figure(figsize=(8, 8))
sns.heatmap(cm1, annot=True,fmt='.20g',cmap='Blues')#添加fmt，不使用科学计数法显示

plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig('confusion matrix-test-SVM.pdf')
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

axes[0].savefig('diabetes-normal-test-ROC-SVM.pdf')