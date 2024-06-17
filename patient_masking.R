
rm(list = ls())

library(tidyr)
library(reshape2)
library(dplyr)

setwd("D:\\Diabetes_data")

a<-read.csv("糖尿病.csv",sep = ",")

a = a[,apply(a, 2, function(x) any(!is.na(x)))]
a<-unique(a)
a<-a[-which(a$检查明细结果 == "不发送"),]

b<-unite(a,"patient",c("姓名","检验时间"), sep="|")

d<-dcast(b,patient+性别+年龄+年龄单位+地址+检查项目+诊断名称~检查明细项目,value.var=c("检查明细结果")) %>%
  separate("patient", into = c("姓名","检验时间"), sep="\\|") %>% 
  group_by(姓名) %>%
  mutate(id = row_number(),.after = "姓名")

patient<-d
patient$id<-paste0(patient$姓名,patient$id)

#################################Data Masking

library(stringr)

index<-which(grepl("之女|之子",d$姓名))

chid<-d[index,]%>% 
  group_by(姓名) %>%
  mutate(id = row_number(),.after = "姓名")
adult<-d[-index,]

max(d$id)#48

ZM<-LETTERS[1:26]
a<-c()
for (i in 1:26) {
  for (j in 1:26) {
    a<-c(a,paste0(ZM[i],ZM[j]))
  }
}

number<-data.frame(c(1:max(d$id)),c(a[1:max(d$id)]))

chid$id<-number[match(chid$id, number[,1]),2]
adult$id<-number[match(adult$id, number[,1]),2]

index1<-intersect(adult$姓名,substr(chid$姓名,1,nchar(chid$姓名)-2))
index2<-which(grepl(paste0(index1,collapse="|"),chid$姓名))
chid1<-chid[index2,]
chid2<-chid[-index2,]

nameA<-data.frame(name=unique(adult$姓名),id=str_pad(c(1:length(unique(adult$姓名))),4, side = "left", "0"))
nameB<-data.frame(name=unique(chid2$姓名),id=c((length(unique(adult$姓名))+1):(length(unique(adult$姓名))+length(unique(chid2$姓名)))))

adult$sample<-paste0("DiabetesA",ifelse(adult$性别=="女","F","M"),nameA[match(adult$姓名, nameA[,1]),2],adult$id)
chid1$sample<-paste0("DiabetesC",ifelse(substr(chid1$姓名,nchar(chid1$姓名)-1,nchar(chid1$姓名))=="之女","F","M"),nameA[match(substr(chid1$姓名,1,nchar(chid1$姓名)-2), nameA[,1]),2],chid1$id)
chid2$sample<-paste0("DiabetesC",ifelse(substr(chid2$姓名,nchar(chid2$姓名)-1,nchar(chid2$姓名))=="之女","F","M"),nameB[match(chid2$姓名, nameB[,1]),2],chid2$id)

patient<-rbind(adult,chid1,chid2) %>%
  select(ncol(adult),1:(ncol(adult)-1),-c("地址","id")) %>% 
  data.frame() %>%
  plyr::rename(c("白细胞"="白细胞数目","单核细胞比率"="单核细胞百分比","单核细胞数"="单核细胞数目","感染红细胞数"="感染红细胞数目",
                 "红细胞"="红细胞数目","红细胞大小CV"="红细胞分布宽度变异系数","红细胞大小SD"="红细胞分布宽度标准差",
                 "红细胞体积M"="平均红细胞体积","淋巴细胞比率"="淋巴细胞百分比","淋巴细胞数"="淋巴细胞数目",
                 "嗜碱性粒细胞比率"="嗜碱性粒细胞百分比","嗜碱性粒细胞数"="嗜碱性粒细胞数目",
                 "嗜酸性粒细胞比率"="嗜酸性粒细胞百分比","嗜酸性粒细胞数"="嗜酸性粒细胞数目",
                 "网织红细胞比率"="网织红细胞百分比","网织红细胞计数"="网织红细胞数目","未成熟粒细胞比率"="未成熟粒细胞百分比",
                 "血红蛋白"="血红蛋白浓度","血红蛋白含量M"="平均红细胞血红蛋白含量","血红蛋白浓度M"="平均红细胞血红蛋白浓度",
                 "血小板"="血小板数目","血小板体积M"="平均血小板体积","血小板体积分布宽度"="血小板分布宽度",
                 "有核红细胞比率"="有核红细胞百分比","有核红细胞计数"="有核红细胞数目","中性细胞比率"="中性粒细胞百分比",
                 "中性细胞数"="中性粒细胞数目"))

sample<-patient[,-2]
patient1<-substr(sample$sample,1,nchar(sample$sample)-2)
sample<-mutate(sample,patient=patient1,.after = "sample")
rownames(sample)<-sample$sample
rownames(patient)<-patient$sample

Diabetes_data<-sample
save(patient,file = "patient.rda")
save(Diabetes_data,file = "Diabetes_data.rda")

Diabetes_chiddata<-Diabetes_data[grepl("DiabetesC",Diabetes_data$sample),]
save(Diabetes_chiddata,file = "Diabetes_chiddata.rda")
