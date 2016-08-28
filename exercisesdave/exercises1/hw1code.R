## EXERCISES 1 ##
library(microbenchmark)
library(ggplot2)
library(Matrix)

# functions
invmethod = function(X,y,W)
{
	return(solve(t(X)%*%W%*%X)%*%t(X)%*%W%*%y)
}

cholmethod = function(X,y,W)
{
	C = t(X)%*%W%*%X
	d = t(X)%*%W%*%y
	L = chol(C)
	z = forwardsolve(L,d)
	b = backsolve(t(L),z)
	return(b)
}

sparsecholmethod = function(X,y,W)
{
	C = t(X)%*%W%*%X
	d = t(X)%*%W%*%y
	L = Cholesky(C,LDL=FALSE)
	z = forwardsolve(L,d)
	b = backsolve(t(L),z)
	return(b)
}

# simulating some silly data
sim = function(N,P)
{
	set.seed(1)
   	beta = rep(1,P)   	X = matrix(rnorm(N*P), nrow=N)
   	y = X%*%beta + rnorm(N,sd=0.05)
   	return(list(y=y,X=X,beta=beta))
}

simsparse = function(N,P)
{
	set.seed(1)
   	beta = rep(1,P)
   	X = matrix(rnorm(N*P), nrow=N)
   	mask = matrix(rbinom(N*P,1,0.05), nrow=N)    X = mask*X
   	y = X%*%beta + rnorm(N,sd=0.05)
   	return(list(y=y,X=X,beta=beta))
}

# testing
N = 2000
P = 200

# X = sim(N,P)$X
# y = sim(N,P)$y
# beta = sim(N,P)$beta

X = simsparse(N,P)$X
y = simsparse(N,P)$y
beta = simsparse(N,P)$beta

res1 = microbenchmark(test1=invmethod(X,y,diag(N)),times=50)
res2 = microbenchmark(test2=cholmethod(X,y,diag(N)),times=50)
X = Matrix(X,sparse=TRUE); W = Matrix(diag(N),sparse=TRUE)
res3 = microbenchmark(test2=cholmethod(X,y,diag(N)),times=50)
print(res1)
print(res2)
print(res3)

test2=cholmethod(X,y,diag(N))

MSE1 = sqrt(sum((test1-beta)^2))
MSE2 = sqrt(sum((test2-beta)^2))

MSE1
MSE2

