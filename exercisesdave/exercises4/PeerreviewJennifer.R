## Comparing Jennifer and David solution paths ##

library(Rcpp)
library(Matrix)
library(RcppEigen)
library(pryr)
library(parallel)
library(doParallel)
library(foreach)
library(MASS)
library(wordspace)
library(spam)

source('bigdatafunctions.R')
sourceCpp('bigdatafunctions.cpp')
sourceCpp('JenniferCpp.cpp')

# read in da big data
X=readRDS('~/Dropbox/PhDCourses/TooBigforGithub/url_X.rds')
varX = colNorms(X)
Xsca = scaleMargins(X,cols=1/varX) #scale just by variance
XJen=t(cbind(rep(1,dim(Xsca)[1]),Xsca))
XDave=as(Xsca,'RsparseMatrix')
y=readRDS('~/Dropbox/PhDCourses/TooBigforGithub/url_y.rds')

# doing the CV
B0 = rep(0,dim(XJen)[1])
lamseq = 1e-10*(1:20)^8.5
prop=0.8
N = length(lamseq)
CVD = list()
CVJ = list()
total = length(y)
trainsize = round(prop*total)
m = rep(1,total)

for(i in 1:N)
{
  CVD[[i]] = davesgdCV(y,XDave,B0,lamseq[i],masterStepSize=1,train=trainsize,npass=10)
  cat(i,'/',N,': Class-error Dave =',CVD[[i]]$Classrate,'\n')
  CVJ[[i]] = sparse_sgd_logit(Xtall=XJen,Y=y,m=m,step=1,train=trainsize,npass=10,beta0=B0,lambda=lamseq[i])
  cat(i,'/',N,': Class-error Jen =',CVJ[[i]]$classrate,'\n')
}
save(CVD,'Dreview.RData')
save(CVJ,'Jreview.RData')

# plotting
load('Dreview.RData')
load('Jreview.RData')
crj = c()
crd = c()
for(i in 1:N)
{
  crj = c(crj,CVJ[[i]]$classrate)
  crd = c(crd,CVD[[i]]$Classrate)
}
indmaxj = which(max(crj)==crj)
indmaxd = which(max(crd)==crd)

plot(crd,type='l',bty='n',xlab=expression(lambda),lwd=2,ylim=c(0,1),xaxt='n',ylab='classification rate')
lines(crj,col='blue',lwd=2)
Axis(labels = signif(1e-10*(1:100)^5,3),at=1:100,side = 1)
points(x=indmaxj,y=crj[indmaxj],pch=19,col='blue')
points(x=indmaxd,y=crd[indmaxd],pch=19,col=1)
legend('bottomleft',legend=c('Jennifer','David'),col=c('blue','black'),lty=1,lwd=2,bty='n')

