################################################################
######     THE C++ implementation stuff HERE!!!    #############
################################################################

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

# running stuff
{

source('bigdatafunctions.R')
sourceCpp('bigdatafunctions.cpp')
sourceCpp('sgdlogit_james.cpp')

# read in da big data
X=readRDS('url_X.rds')
varX = colNorms(X)
Xsca = scaleMargins(X,cols=1/varX) #scale just by variance
X=as(Xsca,'RsparseMatrix')
y=readRDS('url_y.rds')

# # testing out the CV function (works!!)
# Xsmall = X[1:1000,1:1000]
# Xsmall=as(Xsmall,'RsparseMatrix')
# ysmall = y[1:1000]
# B0 = rep(0,dim(Xsmall)[2])
# ptm <- proc.time()
# result = CVfit(ysmall,Xsmall,B0,lambda = 1e-1,masterStepSize = 2,prop = 0.8)
# plot(result$Likelihood,type='l')
# result$Classrate
# result$Specificity
# proc.time() - ptm
# 
# # the full dataset
# B0 = rep(0,dim(X)[2])
# ptm <- proc.time()
# resultdave = CVfit(y,X,B0,lambda = 0.1,masterStepSize = 2,prop = 0.8,npass=5)
# # plot(resultdave$Likelihood,type='l')
# resultdave$alpha
# resultdave$Classrate
# resultdave$Specificity
# resultdave$Sensitivity
# proc.time() - ptm
# # 
# # testing james code
# B0 = rep(0,dim(X)[2])
# result = sparsesgd_logit(t(Xsca),y,rep(1,dim(Xsca)[1]),eta = 2,B0,lambda=1e-8,npass = 1,discount = .001)
# plot(result$nll_tracker,type='l')
# result$alpha
# 
# # compare james and dave
# plot(resultdave$Likelihood,type='l')
# lines(result$nll_tracker,col=rgb(0,1,0,alpha=0.5))
# 
#  
# # # testing the parallel (not working yet)
# # pack = c('MASS','Rcpp','Matrix','RcppEigen','SGD')
# # lamseq = seq(0.5,4,by = 1)
# # N = length(lamseq)
# # cl = makeCluster(1)
# # registerDoParallel(cl)
# # results = foreach(ii=1,.packages=pack)%dopar%CVfit(y,X,B0,1,masterStepSize=1e-2,prop=0.8)
# # stopCluster(cl)

#running individual lambdas
source('bigdatafunctions.R')
sourceCpp('bigdatafunctions.cpp')
B0 = rep(0,dim(X)[2])
lamseq = 1e-10*(1:100)^5
N = length(lamseq)
CVstats = list()
for(i in 1:N)
{
  CVstats[[i]] = CVfit(y,X,B0,lambda = lamseq[i],masterStepSize = 1,prop = 0.8,npass=10)
  cat(i,'/',N,': Class-error =',CVstats[[i]]$Classrate,'\n')
}
save(CVstats,file='CVstatsnewpass=10.RData')

# B0 = rep(0,dim(X)[2])
# lamseq = 1e-4*(1:50)^4
# N = length(lamseq)
# CVstats = list()
# for(i in 1:N)
# {
#   CVstats[[i]] = CVfit(y,X,B0,lambda = lamseq[i],masterStepSize = 1e-2,prop = 0.7)
#   cat(i,'/',N,': Class-error =',CVstats[[i]]$Classrate,'\n')
# }
# save(CVstats,file='CVstats2.RData')


# B0 = rep(0,dim(X)[2])
# lamseq = 1e-5*(1:100)^4
# N = length(lamseq)
# CVstats = list()
# for(i in 1:N)
# {
#   CVstats[[i]] = CVfit(y,X,B0,lambda = lamseq[i],masterStepSize = 1e-2,prop = 0.5)
#   cat(i,'/',N,': Class-error =',CVstats[[i]]$Classrate,'\n')
# }
# save(CVstats,file='CVstats3.RData')

# B0 = rep(0,dim(X)[2])
# lamseq = 1e-5*(1:100)^4
# N = length(lamseq)
# CVstats = list()
# for(i in 1:N)
# {
#   CVstats[[i]] = CVfit(y,X,B0,lambda = lamseq[i],masterStepSize = 1e-2,prop = 0.9)
#   cat(i,'/',N,': Class-error =',CVstats[[i]]$Classrate,'\n')
# }
# save(CVstats,file='CVstats4.RData')

}

# figures
  load('CVstatsnewpass=10.RData')
  classrate = rep(0,100)
  speci = rep(0,100)
  sensi = rep(0,100)
  dist = rep(0,100)
  for(i in 1:100)
  {
    classrate[i] = CVstats[[i]]$Classrate
    speci[i] = CVstats[[i]]$Specificity
    sensi[i] = CVstats[[i]]$Sensitivity
    dist[i] = sqrt( (sensi[i]-1)^2 + (speci[i]-1)^2 )
  }
  indmax = which(max(classrate)==classrate)
  plot(classrate[1:25],type='l',bty='n',xlab=expression(lambda),lwd=2,ylim=c(0,1),xaxt='n',ylab='out of sample statistics')
  Axis(labels = signif(1e-10*(1:100)^5,3),at=1:100,side = 1)
  lines(speci,type='l',col=2,lwd=2)
  lines(sensi,type='l',col=3,lwd=2)
  # lines(1-dist,type='l',col=4,lwd=2)
  points(x=indmax,y=classrate[indmax],pch=19)
  legend('bottomright',legend=c('classrate','specificity','sensitivity'),col=1:3,lty=1,lwd=2,bty='n')

# 
# #   load('CVstats3.RData')
# #   classrate = rep(0,50)
# #   speci = rep(0,50)
# #   sensi = rep(0,50)
# #   dist = rep(0,50)
# #   for(i in 1:50)
# #   {
# #     classrate[i] = CVstats[[i]]$Classrate  
# #     speci[i] = CVstats[[i]]$Specificity
# #     sensi[i] = CVstats[[i]]$Sensitivity
# #     dist[i] = sqrt( (sensi[i]-1)^2 + (speci[i]-1)^2 )
# #   }
# #   indmin = which(min(classrate)==classrate)
# #   plot(classrate,type='l',bty='n',xlab=expression(lambda),lwd=2,col='dark gray',ylim=c(0,1),xaxt='n')
# #   Axis(labels = 1e-4*(1:50)^4,at=1:50,side = 1)
# #   # lines(speci,type='l')
# #   # lines(sensi,type='l')
# #   # lines(dist,type='l')
# #   points(x=41,y=classrate[indmin],pch=19)
# #   
# #   
# #   
# #   
# #   
# }