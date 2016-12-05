# This is the execution file. Creates portfolios and evaluates their performance out-of-sample
# Creates penalized min-variance portfolio, and benchmarks against equal-weighted and minimum variance portfolios. 
# Uses Dow Jones' 30 stocks as assets. 


###
### Preliminaries ------------------------------------------------------
###

# Source functions
source('ADMM.R')

# Daily Realized Covariance matrices
load('DJIA_RKall.RData')

# End of Day Returns
load('DJIAreturns.RData')

### VALUES
number.trading.days = 925 #min(length(COVBIG),dim(Returns)[1])-1

# Form data structures
w = matrix(0,ncol=30,nrow=number.trading.days+1)
wealth <- ew.wealth <- mv.wealth <- numeric(number.trading.days+1)+1
portfolio.return <- ew.return <- mv.return <- numeric(number.trading.days+1)

# Line up Cov's and returns
# Returns: 2007-02-27 until 2010-10-29
# Cov's: 2007-02-01 until 2010-12-31

# These dates should match:
lag = 16 
row.names(Returns)[215]
names(COVBIG)[[232]]
# differences in their indices is 17, but the covariance matrix for time t is used to forecast returns in t+1, so the "lag" is only 16 



###
### Benchmark Portfolios -----------------------------------------
###

### Loop over time
for(t in 1:925){
  Sigma_t = COVBIG[[lag + 3 + t]]
  
  # Equal Weighted
  ew.return[t+1] = mean(as.numeric(Returns[3+t,]) / 100)
  ew.wealth[t+1] = ew.wealth[t] * (1 + ew.return[t+1])
  
  # Unpenalized Minvariance 
  mv.holdings = solve(Sigma_t) %*% rep(1,30)
  mv.holdings = mv.holdings/sum(mv.holdings)
  mv.return[t+1] = sum(mv.holdings * (Returns[3+t,] / 100))
  mv.wealth[t+1] = mv.wealth[t] * (1 + mv.return[t+1])
  
}




###
### "cross validation" to find appropriate lambda ----------------
###

lambda = exp(seq(-22,-4,length.out = 500))
sharpe.ratios = lambda*0

### Initial holdings
avg.cov = (COVBIG[[18]] + COVBIG[[19]])/2
w.hat = avg.cov %*% rep(1,30)
w.hat = w.hat/sum(w.hat)

for(i in 1:length(lambda)){
  
  # Start with fresh data structures
  w = matrix(0,ncol=30,nrow=number.trading.days+1)
  wealth <- numeric(number.trading.days+1)+1
  portfolio.return <- numeric(number.trading.days+1)
  w[1,] = w.hat
  
  # Step through days
  for(t in 1:925){
    Sigma_t = COVBIG[[lag + 3 + t]]
    w[t+1,] = our.portfolio.ADMM(Sigma = Sigma_t, current.holdings = w[t,], 
                                 lambda = lambda[i], 
                                 rho=.1, 
                                 max.iter=100, 
                                 tol = 0.0001,
                                 suppress.output = T)
    portfolio.return[t+1] = sum(w[t+1,] * (Returns[3+t,] / 100))
    #wealth[t+1] =  wealth[t]*(1 + portfolio.return[t+1])
  }
  
  # Calculate Sharpe Ratio
  sharpe.ratios[i] = sqrt(252) * mean(portfolio.return[-1])/sd(portfolio.return[-1])
  print(i/length(lambda))
}


###
### Plot Cross-validation ---------------------------------------------
###

pdf('../crossvalidation.pdf',width=6,height=4)
plot(log(lambda), sharpe.ratios, type='l'
     ,ylab = 'Out-of-sample Sharpe Ratio'
     ,xlab = expression(paste('log(',lambda,')'))
     ,main = "Cross-validation of Penalty Term")

sharpe.mv =  sqrt(252) * mean(mv.return[-1])/sd(mv.return[-1])
points(x = par("usr")[1], y = sharpe.mv, 
       pch = 16, cex=1.3, col = "red", las = 1,
       xpd = TRUE) 

sharpe.ew =  sqrt(252) * mean(ew.return[-1])/sd(ew.return[-1])
points(x = par("usr")[2], y = sharpe.ew, 
       pch = 16, cex=1.3, col = "blue", las = 1,
       xpd = TRUE) 
dev.off()



###
### Portfolio with best Lambda ---------------------------------
###

#_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_#
# lambda in (.00001,.0001) ideally, based on eye test.
lambda = 0.00022 # which is approximately lambda[which.max(sharpe.ratios)]
#_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_/^\_#


# INITIAL HOLDINGS 
### Equal holdings: w[1,] = 1/30
### Use minimum variance of average covariance from first month (feb 2007)
# Would work, but first 17 days are missing 1 of the 30 stocks. 
# Alternate version
avg.cov = (COVBIG[[18]] + COVBIG[[19]])/2
w.hat = avg.cov %*% rep(1,30)
w[1,] = w.hat/sum(w.hat)



### Loop over time - OOS portfolio performance
for(t in 1:925){
	Sigma_t = COVBIG[[lag + 3 + t]]
	w[t+1,] = our.portfolio.ADMM(Sigma = Sigma_t, current.holdings = w[t,], 
	                             lambda = lambda, 
	                             rho=.1, 
	                             max.iter=100, 
	                             tol = 0.0001)
	portfolio.return[t+1] = sum(w[t+1,] * (Returns[3+t,] / 100))
	wealth[t+1] =  wealth[t]*(1 + portfolio.return[t+1])
}







###
### Plot Wealth over time, for best lambda and two benchmark portfolios  -----------------------
###
pdf('../wealth.pdf',height=8,width=6)
par(mfrow=c(2,1))
# Wealth over time
x = seq(2007+2/12,2010+10/12,length.out=length(wealth))
plot(x,wealth,type='l'
     ,ylab='Wealth',xlab='Year',main='Wealth by Strategy'
     ,ylim=range(c(wealth,mv.wealth,ew.wealth)))
lines(x,mv.wealth,col=4)
lines(x,ew.wealth,col=2)
legend('bottomleft',legend=c(expression(paste(lambda,'= 0.00022')),'Equal-Weighted',expression(paste('Unpenalized (',lambda,'=0)',seq=''))),col=c(1,2,4),lty=1,lwd=2)
# Difference in portfolios
plot(x,wealth - ew.wealth,type='l',ylab='Difference',xlab='Year',main='Penalized vs. Equal-Weighted DJ')
#lines(x,mv.wealth - ew.wealth, col=4)
dev.off()


###
### Plot holdings --------------------------
###
pdf('../allholdings.pdf',width=6,height=4)
par(mfrow=c(1,1))
plot(x,w[,1],type='l',ylim=range(w),ylab='Holding Weight')
for(i in 2:30){
  lines(x,w[,i],col=i)
}
title('Strategic Holdings Over Time')
dev.off()


