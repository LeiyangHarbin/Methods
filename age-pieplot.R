
################
rm(list = ls())
setwd("D:/data/diabetes")
diabetes<-read.table(file = 'diabetes.tsv',sep = '\t')
patient<-duplicated(diabetes$patient)
patients<-diabetes[!patient,]

age<-patients[,c(2,5)]
age$group<-cut(age$年龄,breaks = c(0,20,40,60,80,Inf),labels = c('<20','20-40','40-60','60-80','>80'))

age_count<-table(age$group)
counts<-data.frame(age_count)
colnames(counts)<-c('Age','Number')


library(RColorBrewer)  
library(dplyr)
library(graphics)
library(ggrepel)
setwd('D:/data/diabetes/character')
counts<-arrange(counts,Number)

counts$label<-paste0(counts$Age,"\n(",round(counts$Number/sum(counts$Number)*100,2),"%)")
colors <- c("#00B0F6", "#F8766D", "#00BF7D", "#E76BF3", "#A3A500")

pie<-ggplot(counts,aes(x="",y=Number,fill=label))+geom_bar(stat = 'identity',width = 1)+
  coord_polar(theta = "y")+labs(title='',fill="Age")+scale_fill_manual(values = colors)+theme_void()
print(pie)

save(pie,file = 'age-pieplot.rda')
ggsave('age-pieplot.pdf',width = 6,height = 6)
