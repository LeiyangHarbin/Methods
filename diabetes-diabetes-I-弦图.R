
##################
rm(list = ls())
library(statnet)
library(circlize)
setwd('D:/data/diabetes')
matrix<-matrix(nrow = 10,ncol = 2)
colnames(matrix)<-c('Diabetes','Diabetes-II')
rownames(matrix)<-c('MCHC','MCH','MCV','HGB','RBC','PDW','MO','LY','LY%','RDW-SD')
matrix[,1]<-c(1.07,0.73,0.40,0.33,0.35,0.23,0.27,0.23,0.96,0.80)
matrix[,2]<-c(1.21,0.79,0.36,0.43,0.37,0.22,0.44,0.37,1.27,1.02)
matrix<-t(matrix)

grid.col=NULL
grid.col[c('Diabetes','Diabetes-II')]=c('blue','red')

circos.par(gap.degree=c(rep(2,nrow(matrix)-1),10,rep(2,ncol(matrix)-1),10),start.degree=180)
chordDiagram(matrix,directional=F,diffHeight=0.06,grid.col=grid.col,transparency=0.5)

pdf(file = 'circlize-diabetes-all-II.pdf',width = 8,height = 8)
chordDiagram(matrix,directional=F,diffHeight=0.06,grid.col=grid.col,transparency=0.5)
dev.off()
