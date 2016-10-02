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
y=readRDS('url_y.rds')

samps = 100000
Xsam = X[1:samps,]
ysam = y[1:samps]

B0 = rep(0,dim(Xsam)[2])
test=davesgd(ysam,Xsam,B0,lambda = 0,masterStepSize = 1e-1)
plot(test$Likelihood,type='l')
