from __future__ import division
import numpy
import matplotlib.pyplot as plt

def new_R(lam,mu,e,g,sig,b1,b2,n1=1000,n2=1000):
    x=(e+sig+mu)*(b1*n1*sig+b2*n2*sig)
    y=2*e*mu+(mu**2)+2*e*sig+2*sig*mu+(sig**2)
    num=(((x**2)-4*b1*b2*n1*n2*y*(sig**2))**0.5)+x
    den= 2*y*(g+mu)
    return (lam/mu)*(num/den)

tr_val=numpy.linspace(0,1,20)
g=0.1
b1=0.0002
b2=0.0008
sig=0.5
lam=0.0005
mu=0.0002
n1=1000
n2=1000
r1=b1*n1*lam*sig/((mu+g)*mu*(sig+mu))
r2=b2*n2*lam*sig/((mu+g)*mu*(sig+mu))
total=[]
for i in tr_val:
    total.append(new_R(lam,mu,i,g,sig,b1,b2,n1,n2))
plt.plot(tr_val,[r1]*20,'r-')
plt.plot(tr_val,[r2]*20,'k-')
plt.plot(tr_val,[(r1+r2)/2]*20,'g-')
plt.plot(tr_val,total,'bo')
plt.savefig("plot.png")
plt.show()
