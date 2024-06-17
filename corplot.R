
##########
rm(list = ls())
library(corrplot)
setwd('D:/data/diabetes/diabetes-normal')

train<-read.table('diabetes-normal-train.tsv',sep = '\t',header = T,check.names = F)
featurename<-read.table('D:/data/diabetes/血常规指标中英文对照.txt',sep='\t')
colnames(train)<-featurename[match(colnames(train),featurename[,1]),2]
train_data<-train[,which(colnames(train)!=' ')]

cor<-cor(train_data)
corrplot(cor)
pdf('D-N-train-feature-cor.pdf',width = 8,height = 8)
corrplot(cor)
dev.off()
