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

cholmethodgen = function(C,d)
{
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
  -sum( y*log(w+1e-6) + (m-y)*log(1-w+1e-6) )
}

grad = function(y,X,w,m)
{
  -t(X) %*% (y - m*w)	
}

hessian = function(X,m,w)
{
  t(X) %*% diag(m*(w+1e-6)*(1-w+1e-6)) %*% X
}
dist = function(B)
{
  sqrt(sum(B^2))
}

# currently not working ...
newton = function(y,X,B0,m=1,tol,iter,alpha)
{
  p = dim(X)[2]
  N = dim(X)[1]
  Bmat = matrix(0,iter,p)
  Bmat[1,] = B0
  mvec = rep(m,N)
  loglik = rep(0,iter)
  distance = rep(0,iter)
  
  for(ii in 2:iter)
  {
    w = as.numeric(wts(Bmat[ii-1,],X))
    Hess = hessian(X,mvec,w)
    Grad = grad(y,X,w,mvec)
    delB = cholmethodgen(Hess,Grad)  
    Bmat[ii,] = Bmat[ii-1,] - delB
    distance[ii] = dist(Bmat[ii,]-Bmat[ii-1,])
    if(distance[ii] <= tol){ break }
    loglik[ii] = loglike(y,w,m)
  }
  return(list(Bmat=Bmat,loglik=loglik,dist=distance))
}

newtonapprox = function(y,X,B0,m=1,tol,iter,alpha)
{
  p = dim(X)[2]
  N = dim(X)[1]
  Bmat = matrix(0,iter,p)
  Bmat[1,] = B0
  M = diag(rep(m,N))
  distance = rep(0,iter)
  
  for(ii in 2:iter)
  {
    w = as.numeric(wts(Bmat[ii-1,],X))
    Wtil = diag(w*1(1-w))
    S = y - m*w
    A = M %*% Wtil
    z = X %*% Bmat[ii-1] + solve(A) %*% S
    
    
  }
}

steepdescent = function(y,X,B0,m=1,tol,iter,alpha)
{
  p = dim(X)[2]
  N = dim(X)[1]
  Bmat = matrix(0,iter,p)
  Bmat[1,] = B0
  loglik = rep(0,iter)
  distance = rep(0,iter)
  mvec = rep(m,N)
  
  for(ii in 2:iter)
  {
    w = wts(Bmat[ii-1,],X)
    Bmat[ii,] = Bmat[ii-1,] - alpha*grad(y,X,w,mvec)
    distance[ii] = dist(Bmat[ii,]-Bmat[ii-1,])
    if(distance[ii] <= tol){ break }
    loglik[ii] = loglike(y,w,m)
  }
  return(list(Bmat=Bmat,loglik=loglik,dist=distance))
}
  
  
