from __future__ import division
import numpy
import matplotlib.pyplot as plt

def R_e1(e,g,r1,r2):
    num=g*(r1+r2)+(4*e**(2)+(g**2)*((r1-r2)**2))**0.5
    den=2*(e+g)
    return num/den

def R_e2(e,g,r1,r2):
    num=(4*e**(2)+(g**2)*((r1-r2)**2))**0.5
    den=e+(g/2)*(1-((r1+r2)/2))
    return num/den

def R_e3(e,g,r1,r2):
    den=2*(e+g)+(4*e**(2)+(g**2)*((r1-r2)**2))**0.5
    num=g*(r1+r2)
    return (r1+r2)/2+(num/den)

tr_val=numpy.linspace(0,0.1,20)
r1=5
r2=10
g=0.1
total=[]
for i in tr_val:
    #total.append(R_e1(i,g,r1,r2))
    #total.append(R_e2(i,g,r1,r2))
    total.append(R_e3(i,g,r1,r2))
plt.plot(tr_val,[r1]*20,'r-')
plt.plot(tr_val,[r2]*20,'k-')
plt.plot(tr_val,[(r1+r2)/2]*20,'g-')
plt.plot(tr_val,total,'bo')
plt.show()
