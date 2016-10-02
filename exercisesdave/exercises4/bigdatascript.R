################################################################
######     THE C++ implementation stuff HERE!!!    #############
################################################################

library(Rcpp)
library(Matrix)
library(RcppArmadillo)
library(RcppEigen)
library(pryr)
location = "~/Dropbox/PhDCourses/TooBigforGithub/url_svmlight/"
source('bigdatafunctions.R')
sourceCpp('bigdatafunctions.cpp')

# read in da big data
X=readRDS('url_X.rds')
X=as(X,'RsparseMatrix')
y=readRDS('url_y.rds')

# smaller test set
samps = 100000
Xsam = X[1:samps,]
Xsam=as(Xsam,'RsparseMatrix')
ysam = y[1:samps]

sourceCpp('bigdatafunctions.cpp')
B0 = rep(0,dim(Xsam)[2])
test=davesgd(ysam,Xsam,B0,lambda = 1e-5,masterStepSize = 1e-2)
plot(test$Likelihood,type='l',col='black',lwd=1)

# running it on actual data
sourceCpp('bigdatafunctions.cpp')
B0 = rep(0,dim(X)[2])
ptm <- proc.time()
test=davesgd(y,X,B0,lambda = 1e-2,masterStepSize = 1e-2)
proc.time() - ptm
plot(test$Likelihood,type='l',col='black',lwd=1)

# testing out the CV function
results = CVfit(y,X,B0,lambda = 1e-10,masterStepSize = 1e-2,prop = 0.8)
results$Classrate
