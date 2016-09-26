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
  y = X%*%beta + rnorm(N,sd=0.5)
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

# normal loglikelihood with known sigma of 0.25
logliknorm = function(y,X,B)
{
  -sum(dnorm(y,X*B,sd=0.5,log=TRUE))
}

gradnorm = function(y,X,B)
{
  -sum((y-X*B)*X)
}

# stochastic gradient descent
stochgraddescent = function(y,X,B0,m=1,tol,iter,replace)
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
    alpha = rm_step(C=40,a=.5,t=ii,t0=2)
    # alpha=1e-2
    ind = sample(1:N,1)
    ysam = y[ind]
    Xsam = t(as.matrix(X[ind,]))
    msam = mvec[ind]
    wsam = wts(Bmat[ii-1,],Xsam)
    Bmat[ii,] = Bmat[ii-1,] - alpha*grad(ysam,Xsam,wsam,msam)
    distance[ii] = dist(Bmat[ii,]-Bmat[ii-1,])
    w = wts(Bmat[ii-1,],X)
    loglik[ii] = loglike(y,w,m)
  }
  return(list(Bmat=Bmat,loglik=loglik,dist=distance))
}

# stochastic gradient descent
stochgraddescent.test = function(y,X,B0,m=1,tol,iter,alpha,replace)
{
  # p = dim(X)[2]
  # N = dim(X)[1]
  p = 1
  N = 500
  Bmat = matrix(0,iter,p)
  Bmat[1,] = B0
  loglik = rep(0,iter)
  distance = rep(0,iter)
  
  for(ii in 2:iter)
  {
    ind = sample(1:N,1)
    ysam = y[ind]
    Xsam = X[ind]
    Bmat[ii,] = Bmat[ii-1,] - alpha*gradnorm(ysam,Xsam,Bmat[ii-1,])
    distance[ii] = dist(Bmat[ii,]-Bmat[ii-1,])
    # if(distance[ii] <= tol){ break }
    loglik[ii] = logliknorm(y,X,Bmat[ii,])
  }
  return(list(Bmat=Bmat,loglik=loglik,dist=distance))
}

# function borrowed from Jennifer
rm_step <- function(C,a,t,t0)
{
  step <- C*(t+t0)^(-a)
  return(step)
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
    loglik[ii] = loglike(y,w,m)
    Bmat[ii,] = Bmat[ii-1,] - alpha*grad(y,X,w,mvec)
    distance[ii] = dist(Bmat[ii,]-Bmat[ii-1,])
    if(distance[ii] <= tol){ break }
  }
  return(list(Bmat=Bmat,loglik=loglik,dist=distance))
}

# steepest (gradient) descent implementation
steepdescent_backtrack = function(y,X,B0,m=1,tol,iter,alpha,rho,c)
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
    cat(ii,'\n')
    w = wts(Bmat[ii-1,],X)
    loglik[ii] = loglike(y,w,m)
    alphause = backtrack(alpha,rho,c,Bmat[ii-1,],X,y,m)
    Bmat[ii,] = Bmat[ii-1,] - alphause*grad(y,X,w,mvec)
    distance[ii] = dist(Bmat[ii,]-Bmat[ii-1,])
    if(distance[ii] <= tol){ break }
  }
  return(list(Bmat=Bmat,loglik=loglik,dist=distance))
}

# backtracking line search for steepest descent!!!
# Q: can I define new sub-function from a general function?
backtrack = function(a,rho,c,beta,X,y,m)
{
  mvec = rep(m,length(y))
  wold = wts(beta,X)
  direct = grad(y,X,wold,mvec)
  wnew = wts(beta + a*direct,X)
  left = loglike(y,wnew,m)
  right = loglike(y,wold,m) + c*a*t(grad(y,X,wold,mvec)) %*% direct
  while(left > right)
  {
    a = rho*a
    wnew = wts(beta + a*direct,X)
    left = loglike(y,wnew,m)
  }
  return(a)
}


# BFGS Hessian
BFGSinvHess = function(betadiff,graddiff,Hessk)
{
  sy = betadiff %*% t(graddiff)
  ys = graddiff %*% t(betadiff)
  rho = as.numeric(1 / (t(graddiff) %*% betadiff))
  ss = betadiff %*% t(betadiff)
  I = diag(rep(1,length(betadiff)))
  
  Hesskplus1 = (I-rho*sy) %*% Hessk %*% (I-rho*ys) + rho*ss
  # cat(betadiff,'\n')
  return(Hesskplus1)
}

# newton's method optimization implementation 
newton_BFGS_backtrack = function(y,X,B0,m=1,tol,iter,alpha,rho,c)
{
  # defining relevant variables and Bmat
  p = dim(X)[2]
  N = dim(X)[1]
  mvec = rep(m,N)
  loglik = rep(0,iter)
  distance = rep(0,iter)
  Bmat = matrix(0,iter,p)
  Bmat[1,] = B0
  # invHess = diag(p)
  # alphause = backtrack_BFGS(alpha,rho,c,Bmat[1,],X,y,m,invHess)
  # Grad = grad(y,X,wts(Bmat[1,],X),mvec)
  # Bmat[2,] = Bmat[1,] - alphause * (invHess %*% Grad)
  Bmat[2,] = Bmat[1,] - (1e-4)*grad(y,X,wts(Bmat[1,],X),mvec) # gradient descent for second step
  g2 = grad(y,X,wts(Bmat[2,],X),mvec)
  g1 = grad(y,X,wts(Bmat[1,],X),mvec)
  invHess = BFGSinvHess(Bmat[2,]-Bmat[1,],g2-g1,diag(p))
  
  # iteration loop
  for(ii in 3:iter)
  {
    # calculating gradient
    Grad = grad(y,X,wts(Bmat[ii-1,],X),mvec)
    
    # solve linear system for "beta step"
    alphause = backtrack_BFGS(alpha,rho,c,Bmat[ii-1,],X,y,m,invHess)
    Bmat[ii,] = Bmat[ii-1,] - alphause * (invHess %*% Grad)
    
    # compute new Hessian approximation
    g2 = grad(y,X,wts(Bmat[ii,],X),mvec)
    g1 = grad(y,X,wts(Bmat[ii-1,],X),mvec)
    invHess = BFGSinvHess(Bmat[ii,]-Bmat[ii-1,],g2-g1,invHess)
    
    # calculate the step size for convergence, etc.
    distance[ii] = dist(Bmat[ii,]-Bmat[ii-1,])
    # if(distance[ii] <= tol){ break }
    loglik[ii] = loglike(y,wts(Bmat[ii,],X),m)
  }
  return(list(Bmat=Bmat,loglik=loglik,dist=distance))
}

# backtrack for BFGS
backtrack_BFGS = function(a,rho,c,beta,X,y,m,invHess)
{
  mvec = rep(m,length(y))
  wold = wts(beta,X)
  direct = invHess %*% grad(y,X,wold,mvec)
  wnew = wts(beta + a*direct,X)
  left = loglike(y,wnew,m)
  right = loglike(y,wold,m) + c*a*t(grad(y,X,wold,mvec)) %*% direct
  # cat(invHess,'\n')
  # cat(left,'\n')
  # cat(right,'\n')
  while(left > right)
  {
    a = rho*a
    wnew = wts(beta + a*direct,X)
    left = loglike(y,wnew,m)
  }
  return(a)
}












