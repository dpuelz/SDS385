## EXERCISES 1 ##
library(microbenchmark)
library(ggplot2)
library(Matrix)
source('hw2functions.R')

#####################
# LINEAR REGRESSION #
#####################

# testing
N = 500
P = 1

# extracting simulating data from function
X = simsparse(N,P)$X
y = simsparse(N,P)$y
beta = simsparse(N,P)$beta

# benchmarking invesion method
res1 = microbenchmark(test1=invmethod(X,y,diag(N)),times=50)

# benchmarking cholesky method
res2 = microbenchmark(test2=cholmethod(X,y,diag(N)),times=50)
X = Matrix(X,sparse=TRUE); W = Matrix(diag(N),sparse=TRUE)

# benchmarking sparse cholseky method
res3 = microbenchmark(test2=cholmethod(X,y,diag(N)),times=50)

# printing out some results
print(res1)
print(res2)
print(res3)


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

# steepest descent
B0 = rnorm(11)
fit2 = steepdescent(y,X,B0,m=1,tol=1e-6,iter=80000,alpha=1e-2)
tail(fit2$Bmat)

# compare glm and steepest descent
cat(round(fit2$Bmat[20000,],digits=4))
cat(round(Bglm,digits=4))

plot(fit2$loglik,type='l',log='xy')
plot(fit2$dist,type='l',log='xy')

# newton's method
source('hw1functions.R')
B0 = rep(0,11)
fit3 = newton(y,X,B0,m=1,tol=1e-2,iter=10,alpha=1)
cat(fit3$Bmat[10,])
cat(round(Bglm,digits=4))

# newton's method by iteratively re-weighting least squares
source('hw1functions.R')
B0 = rep(0,11)
fit4 = newtonapprox(y,X,B0,m=1,tol=1e-2,iter=10,alpha=1)
cat(fit4$Bmat[10,])
cat(round(Bglm,digits=4))
