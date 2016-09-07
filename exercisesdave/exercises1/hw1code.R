## EXERCISES 1 ##
library(microbenchmark)
library(ggplot2)
library(Matrix)
source('hw1functions.R')

#####################
# LINEAR REGRESSION #
#####################

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


#####################
#       GLM         #
#####################

data = read.csv('wdbc.csv', header = FALSE,row.names = 1)

# construcing y and X
ya = as.character(data[,1])
y = rep(0,length(ya))
y[which(ya=='M')] = 1
X = as.matrix(data[,2:11])
X = scale(X)
X = cbind(rep(1,length(ya)),X)







