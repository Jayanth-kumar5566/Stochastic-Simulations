from __future__ import division
import numpy
import matplotlib.pyplot as plt

def new_R(e,g,b1,b2,n=1000):
    x=(e+g)*(b1+b2)
    y=2*e*g+(g**2)
    num=(((x**2)-4*b1*b2*y)**0.5)+x
    den= 2*y
    return n*(num/den)

tr_val=numpy.linspace(0,0.1,20)
r1=5
r2=10
g=0.1
b1=0.0005
b2=0.0010
total=[]
for i in tr_val:
    total.append(new_R(i,g,b1,b2))
plt.plot(tr_val,[r1]*20,'r-')
plt.plot(tr_val,[r2]*20,'k-')
plt.plot(tr_val,[(r1+r2)/2]*20,'g-')
plt.plot(tr_val,total,'bo')
plt.show()
