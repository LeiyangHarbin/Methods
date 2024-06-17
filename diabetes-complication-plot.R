
################
rm(list = ls())
setwd("D:/data/diabetes")
diabetes<-read.table(file = 'diabetes.tsv',sep = '\t')
patient<-duplicated(diabetes$patient)
patients<-diabetes[!patient,]

type<-patients[,c(2,8)]
type$group<-ifelse(type$诊断名称 %in% c('1型糖尿病','2型糖尿病','糖尿病'),'No complication','Complication')

type1<-data.frame(table(type$group))
colnames(type1)<-c('Type','Number')

library(RColorBrewer)  
library(dplyr)
library(graphics)
library(ggrepel)
setwd('D:/data/diabetes/character')

type1$label<-paste0(type1$Type,'\n(',round(type1$Number/sum(type1$Number)*100,2),'%)')

pie<-ggplot(type1,aes(x="",y=Number,fill=label))+geom_bar(stat = 'identity',width = 1)+
  coord_polar(theta = "y")+labs(title='',fill="Type")+theme_void()
print(pie)

save(pie,file = 'diabetes-compication-plot.rda')
ggsave('diabetes-complication-plot.pdf',width = 6,height = 6)
