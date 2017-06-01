from __future__ import division
import matplotlib.pyplot as plt
import numpy
import r0
file=open("workwith.csv")
a=file.readlines()
a=a[0].strip(',')
a=a.split(',')
count =0
for i in a:
    a[count]=int(i)
    count += 1
gamma=0.4
t1ser=a
#t2ser=I2Val[:count]
plt.plot(t1ser)
plt.show()
from pandas import Series
ser=Series(data=t1ser)#Selects non zero entries
#cut_ser=ser[(ser.T != 0)]
#plt.plot(cut_ser)
#plt.show()
window=5
avg_ser=ser.rolling(window=window).mean()
ts=numpy.log(avg_ser)
X_=ts.to_dict().values()
ind=X_.index(numpy.nanmax(X_))+1
length_of_window=ind	
print "Choose your parts based on this", length_of_window
plt.plot(avg_ser[:ind])
plt.show()
#plt.plot(numpy.log(avg_ser[window-1:30]),'o')

import scipy
r=[]
cor=[]
p=[]
def y_lin(x,slop,lamb):
	y=slop*x+lamb
	return y
x=0
parts=15
while x+parts<=length_of_window :
	ts=numpy.log(avg_ser[int(x):int(x+parts)])
	x=x+1	
	Y=ts.to_dict().values()
	X=ts.to_dict().keys()
#print numpy.corrcoef(X,Y)
	#print "Pearsons r= ",scipy.stats.pearsonr(X,Y)[0]
	#print "P value P= ",scipy.stats.pearsonr(X,Y)[1]
	p.append(scipy.stats.pearsonr(X,Y)[1])
	cor.append(scipy.stats.pearsonr(X,Y)[0])
	#print X
	A=numpy.vstack([X,numpy.ones(len(X))]).T
	slop,lamb=numpy.linalg.lstsq(A,Y)[0]
	#print slop
	#x=numpy.arange(0,30,1)
	#plt.plot(numpy.log(avg_ser[window-1+i:30+i]),'o')
	X=numpy.array(X)
	Y=numpy.array(Y)
	plt.plot(X,Y,'o')
	plt.plot(X,y_lin(X,slop,lamb))
	plt.show()
	tau=1/gamma
	print "the Ro is", 1+slop*tau
	r.append( 1+slop*tau)
fig,ax = plt.subplots(1,2,sharey=True)
ax[0].plot(cor,r,'-o')
ax[1].plot(p,r,'-o')
plt.show()
#print r0.r0(t1ser,5,100,0.2)
