
######################
rm(list = ls())
setwd("D:/data/diabetes")
diabetes_data<-read.csv('diabetes.tsv',sep = '\t')
load('Blood_routine_masking.rda')

setwd('D:/data/diabetes/diabetes-II')
diabetes<-diabetes_data[grep("2型",diabetes_data$诊断名称),]
normal<-na.omit(Blood_routine_masking)
normal<-normal[normal$白细胞数目!='****',]
normal<-normal[sample(nrow(normal),nrow(diabetes)),]

a<-intersect(colnames(diabetes),colnames(normal))
diabetes<-diabetes[,a]
normal<-normal[,a]

#####将特征名换成英文缩写
featurename<-read.table('D:/data/diabetes/血常规指标中英文对照.txt',sep='\t')
colnames(diabetes)<-featurename[match(colnames(diabetes),featurename[,1]),2]
colnames(normal)<-featurename[match(colnames(normal),featurename[,1]),2]
diabetes<-diabetes[,5:ncol(diabetes)]
normal<-normal[,5:ncol(normal)]

#######分离训练集和测试集
normal_index<-sample(1:nrow(normal),size = 0.7*nrow(normal),replace = F)
normal_train<-normal[normal_index,]
normal_test<-normal[-normal_index,]

diabetes_index<-sample(1:nrow(diabetes),size=0.7*nrow(diabetes),replace = F)
diabetes_train<-diabetes[diabetes_index,]
diabetes_test<-diabetes[-diabetes_index,]

#将2型糖尿病的标签设为1，正常设为0
normal_train$label<-0
normal_test$label<-0
diabetes_train$label<-1
diabetes_test$label<-1

train<-rbind(normal_train,diabetes_train)
test<-rbind(normal_test,diabetes_test)

write.table(train,file = 'normal-diabetes-II-train.tsv',sep = '\t')
write.table(test,file = 'normal-diabetes-II-test.tsv',sep = '\t')
