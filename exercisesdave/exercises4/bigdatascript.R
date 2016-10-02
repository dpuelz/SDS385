################################################################
######     THE C++ implementation stuff HERE!!!    #############
################################################################

library(Rcpp)
library(RcppArmadillo)
library(RcppEigen)
library(pryr)
location = "~/Dropbox/PhDCourses/TooBigforGithub/url_svmlight/"
source('bigdatafunctions.R')
sourceCpp('bigdatafunctions.cpp')

# read in da big data
X=readRDS('url_X.rds')
y=readRDS('url_y.rds')
dim(X)
B0 = rep(0,dim(X)[2])
# test the Cpp function
sgd_iteration(10000,y,X,B0)
