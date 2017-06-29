import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import SIR_functions

def dydt2Cities(Y,t,beta1,beta2,gamma,tr12,tr21,lam,mu):
    S1,I1,R1,S2,I2,R2 = Y
    dS1dt = -beta1*S1*I1 +tr21*S2 -tr12*S1 +lam*(S1+I1+R1) -mu*S1
    dI1dt = beta1*S1*I1 - gamma*I1 +tr21*I2 -tr12*I1 -mu*I1
    dR1dt = gamma*I1 + tr21*R2 -tr12*R1 -mu*R1
    dS2dt = -beta2*S2*I2 +tr12*S1 -tr21*S2 +lam*(S2+I2+R2) -mu*S2
    dI2dt = beta2*S2*I2 - gamma*I2 +tr12*I1 -tr21*I2 -mu*I2
    dR2dt = gamma*I2 + tr12*R1 -tr21*R2 -mu*R2
    return np.array([dS1dt,dI1dt,dR1dt,dS2dt,dI2dt,dR2dt])

def Simulate(tmax,beta1,beta2,gamma,tr12,tr21,lam,mu):
 	Y_in = np.array([1000,1,0,1000,1,0])
	t = np.arange(0,tmax,0.01)
	Y = odeint(dydt2Cities,Y_in,t,args=(beta1,beta2,gamma,tr12,tr21,lam,mu,))
        return (t,Y)
'''
tr=np.linspace(0,0.9,20)
labels=['S1','I1','R1','S2','I2','R2']

birth=0.0005
death=0.0002

(t,Y)=Simulate(100,0.0002,0.0008,0.1,0,0,birth,death)
for y ,label in zip(Y.transpose(),labels):	
    plt.plot(t,y,label=label)
plt.legend(loc='best')
plt.show()
'''
    
#---------------Numerically calculating Ro-----------------------------------
labels=['S1','I1','R1','S2','I2','R2']
def fit(ser,time):
    y=SIR_functions.preprocessing(ser)
    x=SIR_functions.finding_point(ser,time,'max')
    plt.plot(time[y:x],np.log(ser[y:x]),'b-',label='orginal series')
    ser=ser[y:x]
    time=time[y:x]
    sl=SIR_functions.Rcode(time,ser)
    plt.plot(time,SIR_functions.y_sl(time,sl[1],sl[0]),'k-',label='R code')
    plt.legend(loc='best')
    plt.show()
    return sl
def R_eff(lam,mu,e,g,b1,b2,n1=1000,n2=1000):
    x=(e+g+mu)*(b1*n1+b2*n2)
    y=2*e*g+(g**2)+2*e*mu+2*g*mu+(mu**2)
    num=(((x**2)-4*b1*b2*n1*n2*y)**0.5)+x
    den= 2*y
    return (num/den)
def R(beta,gamma,mu,lam,N):
    return (beta*N)/(gamma+mu)

e=0.3
beta1=0.0002
beta2=0.0008
N1=1001
N2=1001
mu=0.0003
lam=0.0005
gamma=0.1


print "==============Theoritical calculation====================="
print "City 1 R0", R(beta1,gamma,mu,lam,N1)
print "City 2 R0", R(beta2,gamma,mu,lam,N2)
print "Total R0",  R_eff(lam,mu,e,gamma,beta1,beta2,N1,N2)
print "===============Numerical Calculations===================="
(t,Y)=Simulate(500,beta1,beta2,gamma,e,e,lam,mu)
for y ,label in zip(Y.transpose(),labels):	
    plt.plot(t,y,label=label)
plt.legend(loc='best')
plt.show()
Y=Y.transpose()
ser_t1=Y[1]
ser_t2=Y[4]
ser_tot=ser_t1+ser_t2
zz=fit(ser_t1,t)
print "City 1 R0", 1+(zz[1]/gamma)
xx=fit(ser_t2,t)
print "City 2 R0", 1+(xx[1]/gamma)
yy=fit(ser_tot,t)
print "Total Ro", 1+(yy[1]/gamma)


'''
tr_val=np.linspace(0,1,20)
x=[]
y=[]
z=[]
r1=[]
r2=[]
theory=[]

for (n,m) in zip(tr_val,tr_val):
    print "Transfer value and simulating", n
    (t,Y)=Simulate(500,beta1,beta2,gamma,n,m,lam,mu)
    Y=Y.transpose()
    t1ser = Y[1]
    t2ser = Y[4]
    totser= t1ser+t2ser
    time=t
    xx=R(beta1,gamma,mu,lam,N1)
    yy=R(beta2,gamma,mu,lam,N2)
    zz=fit(totser,time)
    r1_=fit(t1ser,time)
    r2_=fit(t2ser,time)
    x.append(xx)
    y.append(yy)
    z.append(1+(zz[1]/gamma))
    r1.append(1+(r1_[1]/gamma))
    r2.append(1+(r2_[1]/gamma))
    theory.append(R_eff(lam,mu,n,gamma,beta1,beta2,N1,N2))
    
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
'''
