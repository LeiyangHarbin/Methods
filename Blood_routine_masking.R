
rm(list = ls())

library(tidyr)
library(reshape2)
library(dplyr)
library(openxlsx)
library(stringr)

setwd("D:\\blood_routine")

a<-read.csv("血常规1.csv", sep = ",")
b<-dcast(a,样本号+来源+姓名+性别+年龄~检测项目,value.var=c("检测结果")) %>%
  group_by(姓名) %>%
  mutate(id = row_number(),.after = "姓名")

max(b$id)#6

number<-data.frame(c(1:max(b$id)),c(LETTERS[1:max(b$id)]))

b$id<-number[match(b$id, number[,1]),2]
name<-data.frame(name=unique(b$姓名),id=str_pad(c(1:length(unique(b$姓名))),4, side = "left", "0"))

b$sample<-paste0("NORMAL",name[match(b$姓名, name[,1]),2],b$id)
b$patient<-substr(b$sample,1,nchar(b$sample)-1)

patient<-b %>%
  select(c("sample","patient"),ncol(b),1:(ncol(b)-2),-c("id")) %>% data.frame()
rownames(patient)<-patient$sample

Blood_routine_all<-patient
Blood_routine_masking<-Blood_routine_all[,-c(3,5)]

save(Blood_routine_all,file = "Blood_routine_all.rda")
save(Blood_routine_masking,file = "Blood_routine_masking.rda")
