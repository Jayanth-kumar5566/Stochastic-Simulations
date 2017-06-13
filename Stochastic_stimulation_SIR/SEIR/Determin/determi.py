from __future__ import division
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import SEIR_functions

def dydt2Cities(Y,t,beta1,beta2,gamma,sigma,tr12,tr21,lam,mu):
    S1 = Y[0]
    E1 = Y[1]
    I1 = Y[2]
    R1 = Y[3]
    S2 = Y[4]
    E2 = Y[5]
    I2 = Y[6]
    R2 = Y[7]
    dS1dt = -beta1*S1*I1 -tr12*S1 +tr21*S2 +lam*(S1+E1+I1+R1) -mu*S1
    dE1dt =  beta1*S1*I1 -sigma*E1 -tr12*E1 +tr21*E2 -mu*E1
    dI1dt =  sigma*E1 - gamma*I1 -mu*I1
    dR1dt =  gamma*I1 -tr12*R1 +tr21*R2 -mu*R1
    dS2dt = -beta2*S2*I2 -tr21*S2 +tr12*S1 +lam*(S2+E2+I2+R2) -mu*S2
    dE2dt =  beta2*S2*I2 -sigma*E2 -tr21*E2 +tr12*E1 -mu*E2
    dI2dt =  sigma*E2 - gamma*I2 -mu*I2
    dR2dt =  gamma*I2 -tr21*R2 +tr12*R1 -mu*R2
    return np.array([dS1dt,dE1dt,dI1dt,dR1dt,dS2dt,dE2dt,dI2dt,dR2dt])
 
def Simulate(tmax,beta1,beta2,gamma,sigma,tr12,tr21,lam,mu):
 	Y_in = np.array([1000,1,0,0,1000,1,0,0])
	t = np.arange(0,tmax,0.001)
	Y = odeint(dydt2Cities,Y_in,t,args=(beta1,beta2,gamma,sigma,tr12,tr21,lam,mu))
	#print "Ro is ", (beta1*1000)/float(gamma)	
        '''
        fig,ax = plt.subplots()
	ax.plot(t,Y)
	plt.show()
	'''
        return (t,Y)
    
'''
fig,ax=plt.subplots(2,sharex=True)
(t,Y)=Simulate(100,0.0008,0.0020,0.1,0.5,0,0,0.0005,0.0002)
Y=Y.transpose()
ax[0].plot(t,Y[0],label="S1")
ax[0].plot(t,Y[1],label="E1")
ax[0].plot(t,Y[2],label="I1")
ax[0].plot(t,Y[3],label="R1")
ax[1].plot(t,Y[4],label="S2")
ax[1].plot(t,Y[5],label="E2")
ax[1].plot(t,Y[6],label="I2")
ax[1].plot(t,Y[7],label="R2")
ax[1].legend(loc='best')
ax[0].legend(loc='best')
#plt.show()
plt.savefig("test.pdf")
plt.clf()
'''
#------------------- Numerically calculating thr R0------------------

def fit(ser,time):
    y=SEIR_functions.preprocessing(ser)
    x=SEIR_functions.finding_point(ser,time,'max')
    ser=ser[y:x]
    time=time[y:x]
    sl=SEIR_functions.Rcode(time,ser)
    return sl
def new_R(lam,mu,e,g,sig,b1,b2,n1=1000,n2=1000):
    x=(e+sig+mu)*(b1*n1*sig+b2*n2*sig)
    y=2*e*mu+(mu**2)+2*e*sig+2*sig*mu+(sig**2)
    num=(((x**2)-4*b1*b2*n1*n2*y*(sig**2))**0.5)+x
    den= 2*y*(g+mu)
    return (lam/mu)*(num/den)

lam=0.0005
mu=0.0002
beta1=0.0008
beta2=0.0020
gamma=0.1
N1=1000
N2=1000
sigma=0.5

tr_val=np.linspace(0,1,10)
x=[]
y=[]
z=[]
r1=[]
r2=[]
theory=[]

for (n,m) in zip(tr_val,tr_val):
    print "Transfer value and simulating", n
    (t,Y)=Simulate(100,beta1,beta2,gamma,sigma,n,m,lam,mu)
    Y=Y.transpose()
    t1ser = Y[2]
    t2ser = Y[6]
    totser= t1ser+t2ser
    time=t
    xx=(beta1*N1*sigma*lam)/((gamma+mu)*(sigma+mu)*mu)
    yy=(beta2*N2*sigma*lam)/((gamma+mu)*(sigma+mu)*mu)
    zz=fit(totser,time)
    r1_=fit(t1ser,time)
    r2_=fit(t2ser,time)
    x.append(xx)
    y.append(yy)
    z.append(1+(zz[1]/gamma))
    r1.append(1+(r1_[1]/gamma))
    r2.append(1+(r2_[1]/gamma))
    theory.append(new_R(lam,mu,n,gamma,sigma,beta1,beta2,N1,N2))
    
x=np.array(x)
y=np.array(y)
z=np.array(z)

me=(x+y)/2.0
ma=np.maximum.reduce([x,y])
mi=np.minimum.reduce([x,y])

plt.clf()
plt.figure(figsize=(30,15))
plt.plot(tr_val,z,'bo',label='actual total0')
plt.plot(tr_val,theory,'o-',label='Theoritical R0')
plt.plot(tr_val,r1,'k*',label='r1 cal')
plt.plot(tr_val,r2,'mv',label='r2 cal')
plt.plot(tr_val,me,'g-',label='mean of ro')
plt.plot(tr_val,ma,'r-',label='max of the ro')
plt.plot(tr_val,mi,'k-',label='min of the ro')
plt.legend(loc='best')
plt.savefig('graph6.png', format='png', orientation='landscape')
plt.close()
