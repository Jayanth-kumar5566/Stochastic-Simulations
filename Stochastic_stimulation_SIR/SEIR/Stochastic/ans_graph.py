from __future__ import division
import SEIR_functions
import numpy
import  matplotlib.pyplot as plt
from scipy import polyfit
#import sys
#import time


#----------------------------------Averaging of the series-----------------------------------------------------------
def sim_av(beta1,beta2,tr12,tr21):
    series_t1=[]
    series_t2=[]
    series_total=[]
    time=[]
    count=0
    num_iter=0
    while count<=100:
        (t1ser,t2ser,tot,tim)=SEIR_functions.st_sim(beta1,beta2,lam,mu,N1,N2,gamma,sigma,tr12,tr21)
        num_iter += 1
        if numpy.max(t1ser) > 0.1*N1 or numpy.max(t2ser) > 0.1*N2:
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
            
   # print "Averaging the iterations"
    avg_t1=SEIR_functions.avg_ser(series_t1,time)
    avg_t2=SEIR_functions.avg_ser(series_t2,time)
    avg_tot=SEIR_functions.avg_ser(series_total,time)
    tim_t1=avg_t1[1]
    ser_t1=avg_t1[0]
    tim_t2=avg_t2[1]
    ser_t2=avg_t2[0]
    tim_tot=avg_tot[1]
    ser_tot=avg_tot[0]
    print "Number of Iterations",num_iter
    #print "Size of the total series",(sys.getsizeof(series_total)/sys.maxsize)*100
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
    #return [(ser_tot,tim_tot)]
#---------------------------------Fitting--------------------------------------------------------------------------------

#(t1ser,t2ser,tot,tim)=SIR_functions.st_sim(beta1,beta2,N1,N2,mu,gamma,omega,tr12,tr21,alpha)
def fit(ser,tim):
    y=SEIR_functions.preprocessing(ser)
    x=SEIR_functions.finding_point(ser,tim,'max')
    #x2=SEIR_functions.finding_point(ser,tim,'slope')
    #x=max(x1,x2)

    plt.plot(tim[:x],numpy.log(ser[:x]),'b-',label='orginal series')

    ser=ser[y:x]
    time=tim[y:x]

    '''s=SEIR_functions.fit(ser,time)
    plt.plot(time,SEIR_functions.y_sl(time,s[0],s[1]),'r-',label='fit fn')
    print s[0]
    z=SEIR_functions.Fitt(ser,time)
    print z[0]
    plt.plot(time,SEIR_functions.y_sl(time,z[0],z[1]),'g-',label='Fitt fn')'''

    sl=SEIR_functions.Rcode(time,ser)
    #--------------If needed to check fit

    plt.plot(time,SEIR_functions.y_sl(time,sl[1],sl[0]),'k-',label='R code')
    plt.legend(loc='best')
    plt.show()

    #-------------
    return sl
#-------------------------------------------------------------------------------------------
def R_eff(lam,mu,e,g,sig,b1,b2,n1=1000,n2=1000):
    x=(e+sig+mu)*(b1*sig+b2*sig)
    y=2*e*mu+(mu**2)+2*e*sig+2*sig*mu+(sig**2)
    num=(((x**2)-4*b1*b2*y*(sig**2))**0.5)+x
    den= 2*y*(g+mu)
    return (lam/mu)*(num/den)
def R(beta,gamma,sigma,mu,lam,N):
    return (beta*sigma)/((gamma+mu)*(mu+sigma))


e=0
N1 = 1000
N2 = 1000
mu = 0
lam = 0
gamma = 0.1
sigma = 100
tmax = 100
beta1=0.8
beta2=1.5
print "===========Theoritical caluclation====================="
print "City 1 R0", (beta1*sigma)/((gamma+mu)*(sigma+mu))
print "City 2 R0", (beta2*sigma)/((gamma+mu)*(sigma+mu))
#print "Total R0", new_R(lam,mu,e,gamma,sigma,beta1,beta2,N1,N2)
print "===============Numerical Calculations================="
[(ser_t1,tim_t1),(ser_t2,tim_t2),(ser_tot,tim_tot)]=sim_av(beta1,beta2,e,e)
zz=fit(ser_t1,tim_t1)
print "City 1 R0", 1+(zz[1]/gamma)
xx=fit(ser_t2,tim_t2)
print "City 2 R0", 1+(xx[1]/gamma)
yy=fit(ser_tot,tim_tot)
print "Total Ro", 1+(yy[1]/gamma)


'''
tr_val=numpy.linspace(0,0.9,10)
x=[]
y=[]
z=[]
#r1=[]
#r2=[]
beta1=0.8
beta2=1.5
N1 = 1000
N2 = 1000
mu = 0
lam = 0
gamma = 0.1
sigma = 0.5
tmax = 100
for n in tr_val:
    print "Transfer value and simulating", n
    #[(ser_t1,tim_t1),(ser_t2,tim_t2),(ser_tot,tim_tot)]=sim_av(beta1,beta2,n,n)
    [(ser_tot,tim_tot)]=sim_av(beta1,beta2,n,n)
    xx=R(beta1,gamma,sigma,mu,lam,N1)
    yy=R(beta2,gamma,sigma,mu,lam,N2)
    print "Fitting"
    zz=fit(ser_tot,tim_tot)
    x.append(xx)
    y.append(yy)
    z.append(1+(zz[1]/gamma))
    print "Calculated Total", 1+(zz[1]/gamma)
    #r1.append(1+(r1_[1]/gamma))
    #r2.append(1+(r2_[1]/gamma))
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
plt.legend(loc='best')
plt.savefig('graph8_mu0_lam0.png', format='png', orientation='landscape')
plt.close()
'''
