
###############
rm(list = ls())
library(readxl)
setwd("D:/data/diabetes")
diabetes<-read.table(file = 'diabetes.tsv',sep = '\t')
stan<-read_excel('血常规参考正常值.xls')

stan<-cbind(stan,do.call(rbind,strsplit(as.character(stan$`Reference Value`),'-')))#将参考值分为两列

data<-data.frame(matrix(nrow=3,ncol = 23))

colnames(data)<-stan$Feature

rownames(data)<-c('Low','Normal','High')

stan$`1`<-as.numeric(stan$`1`)
stan$`2`<-as.numeric(stan$`2`)

for (i in 1:nrow(stan)){
  data[1,i]<-sum(na.omit(diabetes[,stan[i,2]]<stan[i,5]))
  data[2,i]<-sum(na.omit(diabetes[,stan[i,2]]>=stan[i,5]&diabetes[,stan[i,2]]<=stan[i,6]))
  data[3,i]<-sum(na.omit(diabetes[,stan[i,2]]>stan[i,6]))
}

library(RColorBrewer)  
library(dplyr)
library(graphics)
library(ggplot2)
library(reshape2)
setwd('D:/data/diabetes/character')

data$Level<-rownames(data)

data1<-melt(data,id.vars='Level')


p1<-ggplot(data=data1,aes(variable,value,fill=Level))+
  geom_bar(stat="identity",position="stack", color="black", width=0.7,size=0.25)+
  labs(x='Blood routine',y='Number of samples')+
  theme(
    axis.title=element_text(size=15,face="plain",color="black"),
    axis.text = element_text(size=10,face="plain",color="black"),
    legend.title=element_text(size=16,face="plain",color="black"),
    legend.text = element_text(size=12,face="plain",color="black"),
    legend.background  =element_blank(),
    panel.grid = element_blank(),
    panel.background = element_blank()
    )
p1

save(p1,file = 'feature-variety-diabetes.rda')
ggsave('feature-variety-diabetes.pdf',width = 16,height = 8)
