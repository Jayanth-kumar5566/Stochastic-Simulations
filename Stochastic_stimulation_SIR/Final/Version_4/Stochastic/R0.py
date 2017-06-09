from __future__ import division
import SIR_functions
import numpy
import  matplotlib.pyplot as plt
N1 = 1000
N2 = 1000
mu = 0
gamma = 0.1
omega = 0
tr12 = 0.05
tr21 = 0.05
tmax = 100
alpha = 0
#beta1=0.0002
#beta2=0.0003

#----------------------------------Averaging of the series-----------------------------------------------------------
def sim_av(beta1,beta2):
    series_t1=[]
    series_t2=[]
    series_total=[]
    time=[]
    count=0
    while count<=100:
        (t1ser,t2ser,tot,tim)=SIR_functions.st_sim(beta1,beta2,N1,N2,mu,gamma,omega,tr12,tr21,alpha)
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
#beta_values=[(0.0002,0.0003),(0.0002,0.0004),(0.0002,0.0005),(0.0002,0.0006),(0.0002,0.0007),(0.0002,0.0008),(0.0002,0.0009),(0.0002,0.0010)]

x=[]
y=[]
z=[]
for i in numpy.arange(0.0002,0.0010,0.0001):
    [(ser_t1,tim_t1),(ser_t2,tim_t2),(ser_tot,tim_tot)]=sim_av(0.0002,i)
    xx=fit(ser_t1,tim_t1)
    yy=fit(ser_t2,tim_t2)
    zz=fit(ser_tot,tim_tot)
    x.append(xx[1])
    y.append(yy[1])
    z.append(zz[1])

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('R0 of city 1')
ax.set_ylabel('R0 of city 2')
ax.set_zlabel('total R0')
plt.show()
x=numpy.array(x)
y=numpy.array(y)
z=numpy.array(z)
Xax=abs(x-y)
me=(x+y)/2.0
ma=numpy.maximum.reduce([x,y])
mi=numpy.minimum.reduce([x,y])
plt.plot(Xax,z,'bo',label='actual total0')
plt.plot(Xax,me,'g*',label='mean of ro')
plt.plot(Xax,ma,'r^',label='max of the ro')
plt.plot(Xax,mi,'kv',label='min of the ro')
plt.legend(loc='best')
plt.show()


