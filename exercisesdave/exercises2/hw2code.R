## EXERCISES 1 ##
library(microbenchmark)
library(ggplot2)
library(Matrix)
source('hw2functions.R')

#####################
#  Simulated Data   #
#####################

# testing
N = 500
P = 1

# extracting simulating data from function
X = sim(N,P)$X
y = sim(N,P)$y
beta = sim(N,P)$beta

# fitting via GLM and SGD
fit = glm(y~X,family='gaussian')
Bglm = fit$coefficients

# stochastic gradient descent on test data (single covariate regression)
B0 = -5
fit2 = stochgraddescent.test(y,X,B0,m=1,tol=1e-6,iter=1000,alpha=1e-1,replace=TRUE)
plot(fit2$Bmat[2:1000,],type='l')
abline(h=1,lty=2,col=2)
plot(fit2$loglik[2:1000],type='l')



#####################
#       GLM         #
#####################

# read in data
data = read.csv('wdbc.csv', header = FALSE,row.names = 1)

# construcing y and X (scaling X and adding in intercept column)
ya = as.character(data[,1])
y = rep(0,length(ya))
y[which(ya=='M')] = 1
X = as.matrix(data[,2:11])
X = scale(X)
X = cbind(rep(1,length(ya)),X)

# glm test
fit = glm(y~X-1,family='binomial')
Bglm = fit$coefficients

# SGD descent
set.seed(2)
B0 = Bglm + rnorm(11)
iter=100000
fit3 = stochgraddescent(y,X,B0,m=1,tol=1e-6,iter,replace=FALSE)
tail(fit3$Bmat)

# compare glm and steepest descent
cat(round(fit3$Bmat[iter,],digits=4))
cat(round(Bglm,digits=4))

matplot(fit3$Bmat[(iter/2):iter,],type='l')
abline(h=Bglm,col='gray')
plot(fit3$loglik[2:iter],type='l',log='xy')
plot(fit3$dist[2:iter],type='l')


