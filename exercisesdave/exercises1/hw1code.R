## EXERCISES 1 ##

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

# simulating some silly data
sim = function(N,P)
{
	set.seed(1)
   	beta = rep(1,P)   	X = matrix(rnorm(N*P), nrow=N)
   	y = X%*%beta + rnorm(N,sd=0.05)
   	return(list(y=y,X=X,beta=beta))
}

# testing
N = 2000
P = 5
X = sim(N,P)$X
y = sim(N,P)$y
beta = sim(N,P)$beta

test1=invmethod(X,y,diag(N))
test2=cholmethod(X,y,diag(N))

MSE1 = sqrt(sum((test1-beta)^2))
MSE2 = sqrt(sum((test2-beta)^2))

MSE1
MSE2

