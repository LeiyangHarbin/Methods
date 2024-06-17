
######################
rm(list = ls())
setwd("D:/data/diabetes")
load('Diabetes_data.rda')
load('Blood_routine_masking.rda')


a<-duplicated(Diabetes_data$检验时间)
diabetes_data<-Diabetes_data[which(a==FALSE),]
diabetes_data1<-diabetes_data[which(diabetes_data$年龄单位=='岁'),]


write.table(diabetes_data1,file = 'diabetes.tsv',sep = '\t')

setwd('D:/data/diabetes/diabetes-normal')

b<-intersect(colnames(diabetes_data1),colnames(Blood_routine_masking))

normal<-na.omit(Blood_routine_masking)
normal1<-normal[normal$白细胞数目!='****',]
normal2<-normal1[,b]
normal_index<-sample(1:nrow(normal2),size = 0.7*nrow(normal2),replace = F)
normal_train<-normal2[normal_index,]
normal_test<-normal2[-normal_index,]

diabetes<-diabetes_data1[,b]
diabetes1<-na.omit(diabetes)
diabetes_index<-sample(1:nrow(diabetes1),size=0.7*nrow(diabetes1),replace = F)
diabetes_train<-diabetes1[diabetes_index,]
diabetes_test<-diabetes1[-diabetes_index,]

normal_train$label<-0
normal_test$label<-0
diabetes_train$label<-1
diabetes_test$label<-1
#?????򲡵ı?ǩ??Ϊ1????????Ϊ0

train<-rbind(normal_train,diabetes_train)
test<-rbind(normal_test,diabetes_test)

write.table(train,file = 'diabetes-normal-train.tsv',sep = '\t')
write.table(test,file = 'diabetes-normal-test.tsv',sep = '\t')
