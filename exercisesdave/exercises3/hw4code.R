## EXERCISES 3 ##
library(microbenchmark)
library(ggplot2)
library(Matrix)
source('hw4functions.R')

# read in data
data = read.csv('wdbc.csv', header = FALSE,row.names = 1)

# construcing y and X (scaling X and adding in intercept column)
ya = as.character(data[,1])
y = rep(0,length(ya))
y[which(ya=='M')] = 1
X = as.matrix(data[,2:11])
X = scale(X)
X = cbind(rep(1,length(ya)),X)
p = dim(X)[2]

# glm test
fit = glm(y~X-1,family='binomial')
Bglm = fit$coefficients

# stochastic gradient descent MINIBATCH
source('hw4functions.R')
set.seed(2)
# B0 = Bglm + rnorm(11)
B0= rep(0,p)
iter=10000
fit3 = stochgraddescent_minibatch(y,X,B0,m=1,tol=1e-6,iter,replace=FALSE,rho=0.85,c=0.35)
tail(fit3$Bmat)
plot(fit3$loglik[-1],type='l',log='xy',lwd=2,bty='n',ylab='log-likelihood')
