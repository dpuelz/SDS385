## HW1 functions ##

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
  beta = rep(1,P)
  X = matrix(rnorm(N*P), nrow=N)
  y = X%*%beta + rnorm(N,sd=0.05)
  return(list(y=y,X=X,beta=beta))
}

simsparse = function(N,P)
{
  set.seed(1)
  beta = rep(1,P)
  X = matrix(rnorm(N*P), nrow=N)
  mask = matrix(rbinom(N*P,1,0.05), nrow=N)
  X = mask*X
  y = X%*%beta + rnorm(N,sd=0.05)
  return(list(y=y,X=X,beta=beta))
}

wts = function(B,X)
{
  1 / (1 + exp(-X %*% B))
}

loglike = function(y,w,m)
{
  sum(dbinom(y,m,w+1e-6,log=TRUE))
}

grad = function(y,X,w,m)
{
  t(X) %*% (y - t(m) %*% w)	
}

steepdescent = function(y,X,B0,tol,iter)
{
  p = dim(X)[2]
  N = dim(X)[1]
  Bmat = matrix(0,iter,p)
  Bmat[1,] = B0
  
  
  for(ii in 1:iter)
  {
    
  }
}
  
  
