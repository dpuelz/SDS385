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

# read in da big data
X=readRDS('url_X.rds')
varX = colNorms(X)
Xsca = scaleMargins(X,cols=1/varX) #scale just by variance
# Xsca = cbind(rep(1,dim(X)[1]),Xsca)
X=as(Xsca,'RsparseMatrix')
XC=as(Xsca,'CsparseMatrix')
y=readRDS('url_y.rds')

# testing out the CV function (works!!)
Xsmall = X[1:1000,1:1000]
Xsmall=as(Xsmall,'RsparseMatrix')
ysmall = y[1:1000]
B0 = rep(0,dim(Xsmall)[2])
ptm <- proc.time()
result = CVfit(ysmall,Xsmall,B0,lambda = 1e-1,masterStepSize = 1e-2,prop = 0.8)
plot(result$Likelihood,type='l')
result$Classrate
result$Specificity
proc.time() - ptm

# the full dataset
B0 = rep(0,dim(X)[2])
ptm <- proc.time()
result = CVfit(y,X,B0,lambda = 1e-1,masterStepSize = 1e-2,prop = 0.8)
plot(result$Likelihood,type='l')
result$Classrate
result$Specificity
proc.time() - ptm

 
# # testing the parallel (not working yet)
# pack = c('MASS','Rcpp','Matrix','RcppEigen','SGD')
# lamseq = seq(0.5,4,by = 1)
# N = length(lamseq)
# cl = makeCluster(1)
# registerDoParallel(cl)
# results = foreach(ii=1,.packages=pack)%dopar%CVfit(y,X,B0,1,masterStepSize=1e-2,prop=0.8)
# stopCluster(cl)

# running individual lambdas
# source('bigdatafunctions.R')
# sourceCpp('bigdatafunctions.cpp')
# B0 = rep(0,dim(X)[2])
# lamseq = 1e-4*(1:50)^4
# N = length(lamseq)
# CVstats = list()
# for(i in 1:N)
# {
#   CVstats[[i]] = CVfit(y,X,B0,lambda = lamseq[i],masterStepSize = 1e-2,prop = 0.5)
#   cat(i,'/',N,': Class-error =',CVstats[[i]]$Classrate,'\n')
# }
# save(CVstats,file='CVstats1.RData')

B0 = rep(0,dim(X)[2])
lamseq = 1e-4*(1:50)^4
N = length(lamseq)
CVstats = list()
for(i in 1:N)
{
  CVstats[[i]] = CVfit(y,X,B0,lambda = lamseq[i],masterStepSize = 1e-2,prop = 0.7)
  cat(i,'/',N,': Class-error =',CVstats[[i]]$Classrate,'\n')
}
save(CVstats,file='CVstats2.RData')


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
{
  load('CVstats4.RData')
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
  indmin = which(min(classrate)==classrate)
  plot(1-classrate,type='l',bty='n',xlab=expression(lambda),lwd=2,col='dark gray',ylim=c(0,1),xaxt='n')
  Axis(labels = 1e-5*(1:100)^4,at=1:100,side = 1)
  # lines(speci,type='l')
  # lines(sensi,type='l')
  # lines(dist,type='l')
  points(x=41,y=classrate[indmin],pch=19)
  
  
  load('CVstats3.RData')
  classrate = rep(0,50)
  speci = rep(0,50)
  sensi = rep(0,50)
  dist = rep(0,50)
  for(i in 1:50)
  {
    classrate[i] = CVstats[[i]]$Classrate  
    speci[i] = CVstats[[i]]$Specificity
    sensi[i] = CVstats[[i]]$Sensitivity
    dist[i] = sqrt( (sensi[i]-1)^2 + (speci[i]-1)^2 )
  }
  indmin = which(min(classrate)==classrate)
  plot(classrate,type='l',bty='n',xlab=expression(lambda),lwd=2,col='dark gray',ylim=c(0,1),xaxt='n')
  Axis(labels = 1e-4*(1:50)^4,at=1:50,side = 1)
  # lines(speci,type='l')
  # lines(sensi,type='l')
  # lines(dist,type='l')
  points(x=41,y=classrate[indmin],pch=19)
  
  
  
  
  
}