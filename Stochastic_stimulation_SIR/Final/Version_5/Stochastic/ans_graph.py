from __future__ import division
import SIR_functions
import numpy
import matplotlib as mpl
import  matplotlib.pyplot as plt
from scipy import polyfit
N1 = 1000
N2 = 1000
mu = 0.0001
lam=0.0001
gamma = 0.1
omega = 0
#tr12 = 0
#tr21 = 0
tmax = 1000
alpha = 0
#beta1=0.0002
#beta2=0.0003

#----------------------------------Averaging of the series-----------------------------------------------------------
def sim_av(beta1,beta2,tr12,tr21):
    series_t1=[]
    series_t2=[]
    series_total=[]
    time=[]
    count=0
    while count<=100:
        (t1ser,t2ser,tot,tim)=SIR_functions.st_sim(beta1,beta2,N1,N2,lam,mu,gamma,omega,tr12,tr21,alpha)
        if numpy.max(t1ser) > 0.1*N1 and numpy.max(t2ser) > 0.1*N2:
            series_t1.append(t1ser)
            series_t2.append(t2ser)
            '''
            plt.plot(tim,t1ser)
            plt.plot(tim,t2ser)
            plt.show()
            plt.plot(tim,numpy.log(t1ser))
            plt.plot(tim,numpy.log(t2ser))
            plt.show()
            '''
            series_total.append(tot)
            time.append(tim)
            count += 1


    avg_t1=SIR_functions.avg_ser(series_t1,time)
    avg_t2=SIR_functions.avg_ser(series_t2,time)
    avg_tot=SIR_functions.avg_ser(series_total,time)
    tim_t1=avg_t1[1]
    ser_t1=avg_t1[0]
    tim_t2=avg_t2[1]
    ser_t2=avg_t2[0]
    tim_tot=avg_tot[1]
    ser_tot=avg_tot[0]
    #----If needed to check
    '''
    plt.plot(tim_t1,ser_t1,'g',label='series1')
    plt.plot(tim_t2,ser_t2,'r',label='series2')
    plt.plot(tim_tot,ser_tot,'b',label='total')
    plt.legend(loc='best')
    plt.show()
    '''
    #-------------
    return [(ser_t1,tim_t1),(ser_t2,tim_t2),(ser_tot,tim_tot)]
#---------------------------------Fitting--------------------------------------------------------------------------------

#(t1ser,t2ser,tot,tim)=SIR_functions.st_sim(beta1,beta2,N1,N2,mu,gamma,omega,tr12,tr21,alpha)
def fit(ser,tim):
    y=SIR_functions.preprocessing(ser)
    x=SIR_functions.finding_point(ser,tim,'max')
    #x2=SIR_functions.finding_point(ser,tim,'slope')
    #x=max(x1,x2)
    '''
    plt.plot(tim[:x],numpy.log(ser[:x]),'b-',label='orginal series')
    '''
    ser=ser[y:x]
    time=tim[y:x]

    '''s=SIR_functions.fit(ser,time)
    plt.plot(time,SIR_functions.y_sl(time,s[0],s[1]),'r-',label='fit fn')
    print s[0]
    z=SIR_functions.Fitt(ser,time)
    print z[0]
    plt.plot(time,SIR_functions.y_sl(time,z[0],z[1]),'g-',label='Fitt fn')'''

    sl=SIR_functions.Rcode(time,ser)
    #print sl
    #--------------If needed to check fit
    '''
    plt.plot(time,SIR_functions.y_sl(time,sl[1],sl[0]),'k-',label='R code')
    plt.legend(loc='best')
    plt.show()
    '''
    #-------------
    return sl
#-------------------------------------------------------------------------------------------
'''
#------------------------------------testing theory and simulation--------------------------
e=0
def new_R(lam,mu,e,g,b1,b2,n1=1000,n2=1000):
    x=(e+g+mu)*(b1*n1+b2*n2)
    y=2*e*g+(g**2)+2*e*mu+2*g*mu+(mu**2)
    num=(((x**2)-4*b1*b2*n1*n2*y)**0.5)+x
    den= 2*y
    return (lam/mu)*(num/den)

(beta1,beta2)=(0.0002,0.0008)
print "=============Theoritical calculations================"
print "City 1 Ro", (beta1*N1*lam)/((mu+gamma)*mu)
print "City 2 R0", (beta2*N2*lam)/((mu+gamma)*mu)
print "Total R0",  new_R(lam,mu,e,gamma,beta1,beta2,N1,N2)
print "=============Numerical calculations================"
[(ser_t1,tim_t1),(ser_t2,tim_t2),(ser_tot,tim_tot)]=sim_av(beta1,beta2,e,e)
r1_=fit(ser_t1,tim_t1)
print "City 1 R0", 1+(r1_[1]/gamma)
r2_=fit(ser_t2,tim_t2)
print "City 2 R0", 1+(r2_[1]/gamma)
zz_=fit(ser_tot,tim_tot)
print "Total R0", 1+(zz_[1]/gamma)
'''
def new_R(lam,mu,e,g,b1,b2,n1=1000,n2=1000):
    x=(e+g+mu)*(b1*n1+b2*n2)
    y=2*e*g+(g**2)+2*e*mu+2*g*mu+(mu**2)
    num=(((x**2)-4*b1*b2*n1*n2*y)**0.5)+x
    den= 2*y
    return (lam/mu)*(num/den)

tr_val=numpy.linspace(0,0.9,10)
x=[]
y=[]
z=[]
r1=[]
r2=[]
beta1=0.0002
beta2=0.0008
theory=[]
for (n,m) in zip(tr_val,tr_val):
    [(ser_t1,tim_t1),(ser_t2,tim_t2),(ser_tot,tim_tot)]=sim_av(beta1,beta2,n,m)
    xx=(beta1*N1*lam)/((mu+gamma)*mu)
    yy=(beta2*N2*lam)/((mu+gamma)*mu)
    zz=fit(ser_tot,tim_tot)
    #r1_=fit(ser_t1,tim_t1)
    #r2_=fit(ser_t2,tim_t2)
    x.append(xx)
    y.append(yy)
    z.append(1+(zz[1]/gamma))
    #r1.append(1+(r1_[1]/gamma))
    #r2.append(1+(r2_[1]/gamma))
    theory.append(new_R(lam,mu,n,gamma,beta1,beta2,N1,N2))
    
x=numpy.array(x)
y=numpy.array(y)
z=numpy.array(z)

me=(x+y)/2.0
ma=numpy.maximum.reduce([x,y])
mi=numpy.minimum.reduce([x,y])

plt.clf()
plt.figure(figsize=(30,15))
plt.plot(tr_val,z,'bo',label='actual total0')
#plt.plot(tr_val,r1,'k*',label='r1 cal')
#plt.plot(tr_val,r2,'mv',label='r2 cal')
plt.plot(tr_val,me,'g-',label='mean of ro')
plt.plot(tr_val,ma,'r-',label='max of the ro')
plt.plot(tr_val,mi,'k-',label='min of the ro')
plt.plot(tr_val,theory,'-o',label='Theory')
plt.legend(loc='best')
plt.savefig('plot.png', format='png', orientation='landscape')
plt.close()
