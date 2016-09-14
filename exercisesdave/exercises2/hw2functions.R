## HW1 functions ##

# functions
invmethod = function(X,y,W)
{
  return(solve(t(X)%*%W%*%X)%*%t(X)%*%W%*%y)
}

# solving weighted least squares via cholesky decomposition
cholmethod = function(X,y,W)
{
  C = t(X)%*%W%*%X
  d = t(X)%*%W%*%y
  L = chol(C)
  z = solve(t(L)) %*% d
  b = solve(L) %*% z
  return(b)
}

# solving general linear system ( Cx = d ) via cholesky decomposition
cholmethodgen = function(C,d)
{
  L = chol(C)
  z = solve(t(L)) %*% d
  b = solve(L) %*% z
  return(b)
}

# solving weighted least squares via sparse cholesky decomposition
sparsecholmethod = function(X,y,W)
{
  C = t(X)%*%W%*%X
  d = t(X)%*%W%*%y
  L = Cholesky(C,LDL=FALSE)
  z = solve(t(L)) %*% d
  b = solve(L) %*% z
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

# simulating some sparse silly data
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

# weight function for logistic model
wts = function(B,X)
{
  1 / (1 + exp(-X %*% B))
}

loglike = function(y,w,m)
{
  -sum( y*log(w+1e-6) + (m-y)*log(1-w+1e-6) )
}

# gradient function for logistic likelihood
grad = function(y,X,w,m)
{
  -t(X) %*% (y - m*w)	
}

# hessian function for logistic likelihood
hessian = function(X,m,w)
{
  t(X) %*% diag(m*(w+1e-6)*(1-w+1e-6)) %*% X
}
dist = function(B)
{
  sqrt(sum(B^2))
}

# newton's method optimization implementation 
newton = function(y,X,B0,m=1,tol,iter,alpha)
{
  # defining relevant variables and Bmat
  p = dim(X)[2]
  N = dim(X)[1]
  Bmat = matrix(0,iter,p)
  Bmat[1,] = B0
  mvec = rep(m,N)
  loglik = rep(0,iter)
  distance = rep(0,iter)
  
  # iteration loop
  for(ii in 2:iter)
  {
    w = as.numeric(wts(Bmat[ii-1,],X))
    
    # calculate hessian
    Hess = hessian(X,mvec,w)
    
    # calculation gradient
    Grad = -grad(y,X,w,mvec)
    
    # solve linear system for "beta step"
    delB = cholmethodgen(Hess,Grad)
    Bmat[ii,] = Bmat[ii-1,] + delB
    
    # calculate the step size for convergence, etc.
    distance[ii] = dist(Bmat[ii,]-Bmat[ii-1,])
    if(distance[ii] <= tol){ break }
    loglik[ii] = loglike(y,w,m)
  }
  return(list(Bmat=Bmat,loglik=loglik,dist=distance))
}

# equivalent newton's method implementation using iterative least squares re-weighting
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
    Wtil = diag((w+1e-6)*(1-w+1e-6))
    
    # "residuals"
    S = y - m*w
    
    # working weights
    A = M %*% Wtil
    
    # working responses
    z = X %*% Bmat[ii-1,] + diag(1/diag(A)) %*% S
    Bmat[ii,] = cholmethod(X,z,A)
    
  }
  return(list(Bmat=Bmat))
}

# steepest (gradient) descent implementation
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

# stochastic gradient descent
stochgraddescent = function(y,X,B0,m=1,tol,iter,alpha)
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
  
  
