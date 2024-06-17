
################
rm(list = ls())
setwd("D:/data/diabetes")
diabetes<-read.table(file = 'diabetes.tsv',sep = '\t')
patient<-duplicated(diabetes$patient)
patients<-diabetes[!patient,]

type<-patients[,c(2,8)]
type$group<-ifelse(grepl('1型',type$诊断名称),'Type 1',ifelse(grepl('2型',type$诊断名称),'Type 2','Unlabel'))

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

save(pie,file = 'diabetes-type-plot.rda')
ggsave('diabetes-type-plot.pdf',width = 6,height = 6)
