from __future__ import division
import SEIR_functions
import SIR_functions
import numpy
import  matplotlib.pyplot as plt
from scipy import polyfit
import sys
tmax = 100

#----------------------------------Averaging of the series-----------------------------------------------------------
def sim_av(beta1,beta2,tr12,tr21):
    series_t1=[]
    series_t2=[]
    series_total=[]
    time=[]
    count=0
    num_iter=0
    while count<=50:
        (t1ser,t2ser,tot,tim)=SEIR_functions.st_sim(beta1,beta2,lam,mu,N1,N2,gamma,sigma,tr12,tr21)
        num_iter += 1
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
    #print "Number of Iterations",num_iter

    #print "Averaging the iterations"
    avg_t1=SEIR_functions.avg_ser(series_t1,time)
    avg_t2=SEIR_functions.avg_ser(series_t2,time)
    avg_tot=SEIR_functions.avg_ser(series_total,time)
    tim_t1=avg_t1[1]
    ser_t1=avg_t1[0]
    tim_t2=avg_t2[1]
    ser_t2=avg_t2[0]
    tim_tot=avg_tot[1]
    ser_tot=avg_tot[0]
    #print "Size of the total series",(sys.getsizeof(series_total)/sys.maxsize)*100
    #----If needed to check
    '''
    plt.plot(tim_t1,ser_t1,'g',label='series1')
    plt.plot(tim_t2,ser_t2,'r',label='series2')
    #plt.plot(tim_tot,ser_tot,'b',label='total')
    #plt.xlim(0,100)
    #plt.ylim(0,600)
    plt.legend(loc='best')
    plt.show()
    '''
    #-------------
    return [(ser_t1,tim_t1),(ser_t2,tim_t2),(ser_tot,tim_tot)]
    #return [(ser_tot,tim_tot)]
#--------------------------------------------------------------------------------------------------
def sim_av_1(beta1,beta2,tr12,tr21):
    #series_t1=[]
    #series_t2=[]
    series_total=[]
    time=[]
    count=0
    num_iter=0
    while count<=50:
        (t1ser,t2ser,tot,tim)=SIR_functions.st_sim(beta1,beta2,N1,N2,lam,mu,gamma,0,tr12,tr21,0)
        num_iter += 1
        if numpy.max(t1ser) > 0.1*N1 and numpy.max(t2ser) > 0.1*N2:
            #series_t1.append(t1ser)
            #series_t2.append(t2ser)
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
    print "Number of Iterations",num_iter

    print "Averaging the iterations"
    #avg_t1=SEIR_functions.avg_ser(series_t1,time)
    #avg_t2=SEIR_functions.avg_ser(series_t2,time)
    avg_tot=SEIR_functions.avg_ser(series_total,time)
    #tim_t1=avg_t1[1]
    #ser_t1=avg_t1[0]
    #tim_t2=avg_t2[1]
    #ser_t2=avg_t2[0]
    tim_tot=avg_tot[1]
    ser_tot=avg_tot[0]
    print "Size of the total series",(sys.getsizeof(series_total)/sys.maxsize)*100
    #----If needed to check
    '''
    plt.plot(tim_t1,ser_t1,'g',label='series1')
    plt.plot(tim_t2,ser_t2,'r',label='series2')
    plt.plot(tim_tot,ser_tot,'b',label='total')
    plt.legend(loc='best')
    plt.show()
    '''
    #-------------
    #return [(ser_t1,tim_t1),(ser_t2,tim_t2),(ser_tot,tim_tot)]
    return [(ser_tot,tim_tot)]

#---------------------------------Fitting--------------------------------------------------------------------------------
def fit(ser,tim):
    y=SEIR_functions.preprocessing(ser)
    x=SEIR_functions.finding_point(ser,tim,'max')
    '''
    plt.plot(tim[:x],numpy.log(ser[:x]),'b-',label='orginal series')
    '''
    ser=ser[y:x]
    time=tim[y:x]
    sl=SEIR_functions.Rcode(time,ser)
    #--------------If needed to check fit
    '''
    plt.plot(time,SEIR_functions.y_sl(time,sl[1],sl[0]),'k-',label='R code')
    plt.legend(loc='best')
    plt.show()
    '''
    #-------------
    return sl
#-------------------------------------------------------------------------------------------
def new_R(lam,mu,e,g,sig,b1,b2,n1=1000,n2=1000):
    x=(e+sig+mu)*(b1*sig+b2*sig)
    y=2*e*mu+(mu**2)+2*e*sig+2*sig*mu+(sig**2)
    num=(((x**2)-4*b1*b2*y*(sig**2))**0.5)+x
    den= 2*y*(g+mu)
    return (lam/mu)*(num/den)

e=0
N1 = 1000
N2 = 1000
mu = 0.0004
lam = 0.0005
gamma = 0.1
sigma = 1
tmax = 100
beta1=0.8
beta2=1.5
print "===========Theoritical caluclation====================="
print "City 1 R0", (beta1*sigma)*lam/((gamma+mu)*(sigma+mu)*mu)
print "City 2 R0", (beta2*sigma)*lam/((gamma+mu)*(sigma+mu)*mu)
print "Total R0", new_R(lam,mu,e,gamma,sigma,beta1,beta2,N1,N2)
print "===============Numerical Calculations================="
[(ser_t1,tim_t1),(ser_t2,tim_t2),(ser_tot,tim_tot)]=sim_av(beta1,beta2,e,e)
zz=fit(ser_t1,tim_t1)
print "City 1 R0", 1+(zz[1]/gamma)*numpy.exp(zz[1]*1/sigma)
xx=fit(ser_t2,tim_t2)
print "City 2 R0", 1+(xx[1]/gamma)*numpy.exp(xx[1]*1/sigma)
yy=fit(ser_tot,tim_tot)
print "Total Ro", 1+(yy[1]/gamma)*numpy.exp(yy[1]*1/sigma)

'''


beta1=1.5
beta2=0.8
tr_val=[0.005,0.1,0.2,0.3,0.4,0.5,0.6]
seir=[]
sir=[]
for (n,m) in zip(tr_val,tr_val):
    print "Transfer value and simulating", n
    [(seir_tot,seir_tim_tot)]=sim_av(beta1,beta2,n,m)
    [(sir_tot,sir_tim_tot)]=sim_av_1(beta1,beta2,n,m)
    print "Fitting"
    zz=fit(seir_tot,seir_tim_tot)
    yy=fit(sir_tot,sir_tim_tot)
    seir.append(1+(zz[1]/gamma))
    sir.append(1+(yy[1]/gamma))
    print "SEIR", seir
    print "SIR", sir

plt.clf()
plt.figure(figsize=(30,15))
plt.plot(tr_val,sir,'bo-',label='SIR totalR0')
plt.plot(tr_val,seir,'ro-',label='Total R0 SEIR')
plt.xlabel("Transfer rate")
plt.ylabel("$R_{0}$ Values")
plt.legend(loc='best')
plt.savefig('II.png', format='png', orientation='landscape')
plt.close()
'''
