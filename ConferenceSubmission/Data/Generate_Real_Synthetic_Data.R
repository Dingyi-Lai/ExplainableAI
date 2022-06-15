
rm(list = ls())
setwd("/Users/aubrey/Documents/SHK/Dropbox/Dingyi/Data")
getwd()

# The RealData.RData file contains the real data used in the following
# simulations. Observations with missing values have been removed.
# The last column of each dataframe is the response.

load("RealData.RData")

# Train RF by tuning mtry("max_features" in python)
# Call python

# Generate synthetic data
enlist <- function (...) 
{
  result <- list(...)
  if ((nargs() == 1) & is.character(n <- result[[1]])) {
    result <- as.list(seq(n))
    names(result) <- n
    for (i in n) result[[i]] <- get(i)
  }
  else {
    n <- sys.call()
    n <- as.character(n)[-1]
    if (!is.null(n2 <- names(result))) {
      which <- n2 != ""
      n[which] <- n2[which]
    }
    names(result) <- n
  }
  result
}

# Simulating data from the MARS model, which contains 
# interaction terms.
sim.mars1 <- function(n,nval,sigma){
  x <- matrix(runif(5*n,0,1),n,5)
  xval <- matrix(runif(5*nval,0,1),nval,5)
  e <- matrix(rnorm(n,0,sigma),n,1)
  eval <- matrix(rnorm(nval,0,sigma),nval,1)
  mu <- as.vector(10*sin(pi*x[,1]*x[,2])+20*(x[,3]-0.05)^2 + 10*x[,4]+5*x[,5])
  muval <- as.vector(10*sin(pi*xval[,1]*xval[,2])+20*(xval[,3]-0.05)^2 + 10*xval[,4]+5*xval[,5])
  y <- as.vector(10*sin(pi*x[,1]*x[,2])+20*(x[,3]-0.05)^2 + 10*x[,4]+5*x[,5]+e)
  yval <- as.vector(10*sin(pi*xval[,1]*xval[,2])+20*(xval[,3]-0.05)^2 + 10*xval[,4]+5*xval[,5]+eval)
  enlist(x,y,xval,yval,mu,muval)
}

# change interaction term $\sin(\pi X_1 X_2)$ to be purely additive
sim.mars2 <- function(n,nval,sigma){
  x <- matrix(runif(5*n,0,1),n,5)
  xval <- matrix(runif(5*nval,0,1),nval,5)
  e <- matrix(rnorm(n,0,sigma),n,1)
  eval <- matrix(rnorm(nval,0,sigma),nval,1)
  mu <- as.vector(10*(sin(pi*x[,1])+sin(pi*x[,2]))+20*(x[,3]-0.05)^2 + 10*x[,4]+5*x[,5])
  muval <- as.vector(10*(sin(pi*xval[,1])+sin(pi*xval[,2]))+20*(xval[,3]-0.05)^2 + 10*xval[,4]+5*xval[,5])
  y <- as.vector(10*(sin(pi*x[,1])+sin(pi*x[,2]))+20*(x[,3]-0.05)^2 + 10*x[,4]+5*x[,5]+e)
  yval <- as.vector(10*(sin(pi*xval[,1])+sin(pi*xval[,2]))+20*(xval[,3]-0.05)^2 + 10*xval[,4]+5*xval[,5]+eval)
  enlist(x,y,xval,yval,mu,muval)
}


# Variance of the signal in MARS model is given here.
v.mars  <- 50.82657  # MARS model

set.seed(1)
n       <- 500   # training size
p       <- 5     # feature dimension
nval    <- 1000   # size of noise = training size

snr     <- 3.52   # signal-noise-ratio # seems like that we have to set that

sigma   <- (v.mars/snr)^0.5 # variance of sampling the noise
xy.obj1 <- sim.mars1(n,nval,sigma) # a list from MARS
x1       <- xy.obj1$x # x in MARS
# xval1    <- xy.obj1$xval # noise from x in MARS
y1       <- xy.obj1$y # y in MARS
# yval1    <- xy.obj1$yval # noise from y in MARS

# mu1      <- xy.obj1$mu # y-epsilon

# maxnd   <- ceiling(seq(2,n/5,length.out = 9)) 
# maxnodes: the maximum number of terminal nodes that any
# tree within the forest can have

# additive
xy.obj2 <- sim.mars2(n,nval,sigma) # a list from MARS
x2       <- xy.obj2$x # x in MARS
y2       <- xy.obj2$y # y

synthetic1 <- data.frame(x1,y1)
synthetic2 <- data.frame(x2,y2)
# Save multiple objects
save(abalone, bike, boston, concrete, cpu, csm, fb, parkinsons,
     servo, solar, synthetic1, synthetic2,
     file = "/Users/aubrey/Documents/SHK/Dropbox/Dingyi/Data/SRData.RData")

# Check
rm(list = ls())
load("SRData.RData")
