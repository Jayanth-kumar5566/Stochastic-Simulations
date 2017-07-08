from __future__ import division
import numpy
import matplotlib.pyplot as plt

def new_R(lam,mu,e,g,b1,b2,n1=1000,n2=1000):
    x=(e+g+mu)*(b1+b2)
    y=2*e*g+(g**2)+2*e*mu+2*g*mu+(mu**2)
    num=(((x**2)-4*b1*b2*y)**0.5)+x
    den= 2*y
    return (num/den)

tr_val=numpy.linspace(0,0.8,2000)
g=0.1
b1=0.8
b2=1.5
lam=1e-7
mu=1e-7
n1=1000
n2=1000
r1=b1*lam/((mu+g)*mu)
r2=b2*lam/((mu+g)*mu)
total=[]
for i in tr_val:
    total.append(new_R(lam,mu,i,g,b1,b2,n1,n2))
plt.ylim(7,16)
plt.xlabel('$\epsilon$',fontsize=15)
plt.ylabel('Effective $R_{0}$',fontsize=15)
plt.title("Theoritical\n"+'$\gamma=0.1$,'+r'$\beta_{1}=0.8$, $\beta_{2}=1.5$'+',$N1=N2=1000$,$\lambda=\mu=10^{-7}$',fontsize=20)
plt.plot(tr_val,[r1]*2000,'r-')
plt.plot(tr_val,[r2]*2000,'k-')
plt.plot(tr_val,[(r1+r2)/2]*2000,'g-')
plt.plot(tr_val,total,'b-')
plt.text(0.7,14.8,'$R_{0}$ of settlement1',fontsize=15)
plt.text(0.7,7.8,'$R_{0}$ of settlement2',fontsize=15)
plt.text(0.7,11.3,'Average of $R_{0}$ of settlements',fontsize=15)
#plt.savefig("plot.png")
plt.show()
