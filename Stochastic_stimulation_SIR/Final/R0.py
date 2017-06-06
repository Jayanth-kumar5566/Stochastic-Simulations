import SIR_functions
import numpy
import  matplotlib.pyplot as plt
N1 = 1000
N2 = 1000
mu = 0
gamma = 0.1
omega = 0
tr12 = 0
tr21 = 0
tmax = 100
alpha = 0
beta1=0.0002
beta2=0.0003

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
    plt.plot(tim_t1,ser_t1,'g',label='series1')
    plt.plot(tim_t2,ser_t2,'r',label='series2')
    plt.plot(tim_tot,ser_tot,'b',label='total')
    plt.legend(loc='best')
    plt.show()
    return [(ser_t1,tim_t1),(ser_t2,tim_t2),(ser_tot,tim_tot)]
#---------------------------------Fitting--------------------------------------------------------------------------------

#(t1ser,t2ser,tot,tim)=SIR_functions.st_sim(beta1,beta2,N1,N2,mu,gamma,omega,tr12,tr21,alpha)
def fit(ser,tim):
    y=SIR_functions.preprocessing(ser)
    x=SIR_functions.finding_point(ser,tim,'max')
    #x2=SIR_functions.finding_point(ser,tim,'slope')
    #x=max(x1,x2)
    plt.plot(tim[:x],numpy.log(ser[:x]),'b-',label='orginal series')
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
    plt.plot(time,SIR_functions.y_sl(time,sl[1],sl[0]),'k-',label='R code')
    plt.legend(loc='best')
    plt.show()
    return sl

[(ser_t1,tim_t1),(ser_t2,tim_t2),(ser_tot,tim_tot)]=sim_av(beta1,beta2)
x=fit(ser_t1,tim_t1)
y=fit(ser_t2,tim_t2)
z=fit(ser_tot,tim_tot)
print(x[1],y[1],z[1])
#----------------------------------------------------------------------------------------
#                            The Ro dependencies
'''
beta_values=[(0.00009,0.00009),(0.00008,0.00008),(0.00007,0.00007),(0.00006,0.00006),(0.00005,0.00005)]
ro=[]
for (i,j) in beta_values:
    (t1ser,t2ser,tot,tim)=SIR_functions.st_sim(i,j,N1,N2,mu,gamma,omega,tr12,tr21,alpha)
    r=[]
    for k in [t1ser,t2ser,tot]:
        y=SIR_functions.preprocessing(k)
        x=SIR_functions.finding_point(k,tim,'max')
        k_ser=k[y:x]
        time=tim[y:x]
        #s=SIR_functions.fit(k_ser,time)
        s=SIR_functions.Rcode(time,k_ser)
        r.append(s)
    ro.append(r)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x=[]
y=[]
z=[]

for i in ro:
    x.append(i[0])
    y.append(i[1])
    z.append(i[2])

ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('R0 of city 1')
ax.set_ylabel('R0 of city 2')
ax.set_zlabel('total R0')

plt.show()


#-----------------------------------------r0 value dependeces----------------------------------------

def Mean(z):
    return (z[0]+z[1])/2
def Max(z):
    return max(z[0],z[1])
def Min(z):
    return min(z[0],z[1])
def X(z):
    return abs(z[0]-z[1])

me=[]
ma=[]
mi=[]
Xax=[]
y=[]

for i in ro:
    me.append(Mean(i))
    ma.append(Max(i))
    mi.append(Min(i))
    Xax.append(X(i))
y=z


plt.plot(Xax,y,'bo',label='actual total0')
plt.plot(Xax,me,'g*',label='mean of ro')
plt.plot(Xax,ma,'r^',label='max of the ro')
plt.plot(Xax,mi,'ko',label='min of the ro')
plt.legend(loc='best')
plt.show()

#-----------------------------------------------------------------------------------------------------------

'''
