x_ <- read.csv("x.csv")
y_ <- read.csv("y.csv")
x<-x_$X0
y<-log(y_$X0)
#plot(x,y)
NQ <- diff(y)/diff(x)
#plot.ts(NQ)
log.NQ <- log(NQ)
#plot.ts(log.NQ)
low <- lowess(log.NQ) #statsmodels.nonparametric.smoothers_lowess
#plot(low)
cutoff <- 0.75
q <- quantile(low$y, cutoff,na.rm=TRUE)#scipy.stats.mstats.mquatiles
#plot.ts(log.NQ)
#abline(h=q)
x.lower <- x[min(which(low$y > q))]
x.upper <- x[max(which(low$y > q))]
#plot(x,y)
#abline(v=c(x.lower, x.upper))
i=which(x.upper == x)
j=which(x.lower == x)
y_new=y[j:i]
x_new=x[j:i]
plot(x_new,y_new)
fit <- lm(y_new ~ x_new)   # y 'as a linear function of' x
abline(fit) 
sl=fit$coefficients
write(sl,'tmp')
print(sl)
