
###############
rm(list = ls())
load("D:/data/diabetes/Diabetes_list.rda")
val<-Diabetes_list$Blood_routine
rownames(val)<-val$序号
setwd("D:/data/diabetes/diabetes-normal")

featurename<-read.table('D:/data/diabetes/血常规指标中英文对照.txt',sep='\t')
colnames(val)<-featurename[match(colnames(val),featurename[,1]),1]
val<-val[,which(colnames(val)!='')]
val<-val[,featurename[,1]]
val<-data.frame(lapply(val,as.numeric))
#####红细胞压积的数值过小，应该是未用百分数表示，这里还原为百分数
val$红细胞压积<-val$红细胞压积*100

val$label<-1
write.table(val,file = 'diabetes-validation.tsv',sep = '\t')
