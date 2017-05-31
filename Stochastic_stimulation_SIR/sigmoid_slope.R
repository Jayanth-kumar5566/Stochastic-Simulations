set.seed(100)
x_ <- read.csv("Stochastic-Simulations/Stochastic_stimulation_SIR/x.csv")
y_ <- read.csv("Stochastic-Simulations/Stochastic_stimulation_SIR/y.csv")
x<-x_$X0
y<-log(y_$X0)
plot(x,y)
NQ <- diff(y)/diff(x)
plot.ts(NQ)
log.NQ <- log(NQ)
plot.ts(log.NQ)
low <- lowess(log.NQ) #statsmodels.nonparametric.smoothers_lowess
plot(low)
cutoff <- 0.75
q <- quantile(low$y, cutoff)#scipy.stats.mstats.mquatiles
plot.ts(log.NQ)
abline(h=q)
x.lower <- x[min(which(low$y > q))]
x.upper <- x[max(which(low$y > q))]
plot(x,y)
abline(v=c(x.lower, x.upper))
