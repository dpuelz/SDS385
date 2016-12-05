euclidean.norm = function(x){
  if(!is.vector(x)){stop('Input value not a vector')}
  sqrt(sum((x)^2))
}

our.portfolio.ADMM = function(Sigma,current.holdings,lambda = 1,rho=.01, max.iter=1000,tol = 0.1, epsilon_r, epsilon_s, suppress.output = FALSE){
  # This solves a constrained, minimum variance portfolio: 
  # \min_{\bm{w}_t}  \frac{1}{2} \bm{w}_t^\prime \Sigma_t \bm{w}_t + \lambda || \bm{w}_t - \bm{w}_{t-1} ||_1 \\
  # \text{subject to}  \bm{w}_t^\prime \bm{1} = 1.
  
  ### CHOICES
  # lambda is L1 penalty constant ~ trading penalty
  # rho is the constant for the augmented Lagrangian (for ADMM mechanics)
  
  
  ### Data dimensions
  nn = dim(Sigma)[2]
  # spot check!
  if(length(current.holdings) != nn){stop('Sigma, current.holdings are not of the same dimension')}
  
  ### Needed structures
  w0 = current.holdings # \tilde{w} above, but w0 is much shorter
  w = numeric(nn)
  z = numeric(nn)
  z_old = numeric(nn)
  s = numeric(nn)
  y = numeric(nn+1)
  r = numeric(nn+1)
  if(missing(epsilon_s)){epsilon_s = tol}
  if(missing(epsilon_r)){epsilon_r = tol}
  


  ### ADMM!
  for(iter in 1:max.iter){
    
    # Update w
    w = - solve(Sigma + rho * (diag(nn) + 1), y[1] + y[-1] - rho*(z + w0 +1))
    
    # Update z
    inner = y[-1]/rho + w - w0 
    temp = abs(inner) - lambda/rho
    z = sign(inner)*ifelse(temp>0,temp,0)
    
    # Calculate residuals - would do this after y, but they don't depend on y, while y uses the same calculation for r
    r = c(sum(w) -1, w - z - w0)
    s = - rho * (z - z_old)

    # Update y
    y = y + rho * r
    
    # Check for convergence
    norm_r = euclidean.norm(r)
    norm_s = euclidean.norm(s)
    if(norm_r < epsilon_r & norm_s < epsilon_s){
      if(!suppress.output){print(paste(iter,'iterations'))}
      return(w)
    }
	
    # correct stepsize
    if(norm_r >= 5*norm_s){
      rho = rho * 2
      y = y / 2
    }
    if(norm_s >= 5*norm_r){
      rho = rho / 2
      y = y * 2
    }

  }
  
  # If failed to converge
  if(!suppress.output){print(paste('Failed to Converge, norm_r =',norm_r,'norm_s =',norm_s))}

  return(w)
  
}

