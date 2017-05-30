from __future__ import division
import matplotlib.pyplot as plt
import numpy
import scipy.stats as ss
from pandas import Series
def y_n(x,slope,lamb):
	return slope*x+lamb
def r0(t1ser,window_mv_avg,gamma):#,parts=30):
	''' t1ser: is a list of time series
	    window: the time window for moving average
	    length_of_window: total length of series to be considered(calculated by the code below)
	    parts= length of the parts to be considered
	    origin: origin of the series window'''
	r=[]
	cor=[]
	p=[]
	ser=Series(data=t1ser)
	avg_ser=ser.rolling(window=window_mv_avg).mean()
	ts=numpy.log(avg_ser)
	X_=ts.to_dict().values()
	ind=X_.index(numpy.nanmax(X_))+1
	length_of_window=ind	
	#print "Choose your parts based on this", length_of_window
	x=0
	parts=int(ind/2)
	while x+parts<=length_of_window :
		ts=numpy.log(avg_ser[int(x):int(x+parts)])
		x=x+1
		#print len(ts)
		Y=ts.to_dict().values()
		X=ts.to_dict().keys()
		p_r=ss.pearsonr(X,Y)[0]
		cor.append(p_r)
		p_p=ss.pearsonr(X,Y)[1]
		p.append(p_p)
		A=numpy.vstack([X,numpy.ones(len(X))]).T
		slop,lamb=numpy.linalg.lstsq(A,Y)[0]
		tau=1/gamma
		ro= 1+slop*tau
		r.append(ro)
		''' #------Plotting to check the fitting-----------------
		plt.plot(X,Y,'o')
		plt.plot(X,map(y_n,X,[slop]*len(X),[lamb]*len(X)))
		plt.show()'''
	#fig,ax = plt.subplots(1,2,sharey=True)
	#ax[0].plot(cor,r)
	#ax[1].plot(p,r)
	#plt.plot()
	return(r,cor,p)
