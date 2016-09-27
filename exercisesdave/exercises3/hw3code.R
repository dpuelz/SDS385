## EXERCISES 3 ##
library(microbenchmark)
library(ggplot2)
library(Matrix)
source('hw3functions.R')

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

# steepest descent
B0 = rep(0,p)
# set.seed(1)
# B0 = Bglm + 0.5*rnorm(p)
fit2 = steepdescent(y,X,B0,m=1,tol=1e-6,iter=500,alpha=1e-2)
tail(fit2$Bmat)

# steepest descent with backtracking
B0 = rep(0,p)
# set.seed(1)
# B0 = Bglm + 0.5*rnorm(p)
fit3 = steepdescent_backtrack(y,X,B0,m=1,tol=1e-6,iter=500,alpha=1,rho=0.9,c=0.01)
tail(fit3$Bmat)
plot(fit3$loglik,type='l',log='xy')
plot(fit3$alphastore[-1],type='l')

# newton method BFGS
source('hw3functions.R')
B0 = rep(0,p)
# set.seed(1)
# B0 = Bglm + 0.5*rnorm(p)
fit4 = newton_BFGS_backtrack(y,X,B0,m=1,tol=1e-6,iter=500,alpha=0.2,rho=0.85,c=0.35)
tail(fit4$Bmat)
plot(fit4$loglik,type='l',log='xy')

# compare glm and steepest descent algorithms
cat(round(fit2$Bmat[20000,],digits=4))
cat(round(fit3$Bmat[20000,],digits=4))
cat(round(Bglm,digits=4))

# compare convergence
plot(fit2$loglik[-1],type='l',log='xy',lwd=2,bty='n',ylab='log-likelihood')
lines(fit3$loglik[-1],type='l',col='blue',lwd=2)
lines(fit4$loglik[c(-1,-2)],type='l',col='red',lwd=2)
legend('topright',legend=c('GD','GD with linesearch','Quasi-Newton with linesearch'),lty=1,lwd=2,bty='n',col=c(1,'blue','red'))

plot(fit2$dist,type='l',log='xy')
lines(fit3$dist,type='l',log='xy',col='blue')
