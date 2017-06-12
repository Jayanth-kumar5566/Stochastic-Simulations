from __future__ import division
import SEIR_functions
import numpy
import  matplotlib.pyplot as plt
from scipy import polyfit
N1 = 1000
N2 = 1000
gamma = 0.1
sigma = 0.5
#tr12 = 0
#tr21 = 0
tmax = 100
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
        (t1ser,t2ser,tot,tim)=SEIR_functions.st_sim(beta1,beta2,N1,N2,gamma,sigma,tr12,tr21)
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


    avg_t1=SEIR_functions.avg_ser(series_t1,time)
    avg_t2=SEIR_functions.avg_ser(series_t2,time)
    avg_tot=SEIR_functions.avg_ser(series_total,time)
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
    y=SEIR_functions.preprocessing(ser)
    x=SEIR_functions.finding_point(ser,tim,'max')
    #x2=SEIR_functions.finding_point(ser,tim,'slope')
    #x=max(x1,x2)
    '''
    plt.plot(tim[:x],numpy.log(ser[:x]),'b-',label='orginal series')
    '''
    ser=ser[y:x]
    time=tim[y:x]

    '''s=SEIR_functions.fit(ser,time)
    plt.plot(time,SEIR_functions.y_sl(time,s[0],s[1]),'r-',label='fit fn')
    print s[0]
    z=SEIR_functions.Fitt(ser,time)
    print z[0]
    plt.plot(time,SEIR_functions.y_sl(time,z[0],z[1]),'g-',label='Fitt fn')'''

    sl=SEIR_functions.Rcode(time,ser)
    #print sl
    #--------------If needed to check fit
    '''
    plt.plot(time,SEIR_functions.y_sl(time,sl[1],sl[0]),'k-',label='R code')
    plt.legend(loc='best')
    plt.show()
    '''
    #-------------
    return sl
#-------------------------------------------------------------------------------------------
#beta_values=[(0.0002,0.0003),(0.0002,0.0004),(0.0002,0.0005),(0.0002,0.0006),(0.0002,0.0007),(0.0002,0.0008),(0.0002,0.0009),(0.0002,0.0010)]
(beta1,beta2)=(0.0015,0.0100)
[(ser_t1,tim_t1),(ser_t2,tim_t2),(ser_tot,tim_tot)]=sim_av(beta1,beta2,0,0)
print "City 1", (beta1*N1)/gamma
print "City 2", (beta2*N2)/gamma
zz=fit(ser_t1,tim_t1)
xx=fit(ser_t2,tim_t2)
print "calculated City 1", 1+(zz[1]/gamma)
print "calculated City 2", 1+(xx[1]/gamma)

'''
tr_val=numpy.linspace(0,1,10)
x=[]
y=[]
z=[]
r1=[]
r2=[]
for (n,m) in zip(tr_val,tr_val):
    beta1=0.0015
    beta2=0.0045
    [(ser_t1,tim_t1),(ser_t2,tim_t2),(ser_tot,tim_tot)]=sim_av(beta1,beta2,n,m)
    xx=(sigma*beta1*N1)/gamma
    yy=(sigma*beta2*N2)/gamma
    zz=fit(ser_tot,tim_tot)
    r1_=fit(ser_t1,tim_t1)
    r2_=fit(ser_t2,tim_t2)
    x.append(xx)
    y.append(yy)
    z.append(1+(zz[1]/gamma))
    r1.append(1+(r1_[1]/gamma))
    r2.append(1+(r2_[1]/gamma))
    
x=numpy.array(x)
y=numpy.array(y)
z=numpy.array(z)

me=(x+y)/2.0
ma=numpy.maximum.reduce([x,y])
mi=numpy.minimum.reduce([x,y])

plt.clf()
plt.figure(figsize=(30,15))
plt.plot(tr_val,z,'bo',label='actual total0')
plt.plot(tr_val,r1,'k*',label='r1 cal')
plt.plot(tr_val,r2,'mv',label='r2 cal')
plt.plot(tr_val,me,'g-',label='mean of ro')
plt.plot(tr_val,ma,'r-',label='max of the ro')
plt.plot(tr_val,mi,'k-',label='min of the ro')
plt.legend(loc='best')
plt.savefig('graph4.png', format='png', orientation='landscape')
plt.close()
'''
