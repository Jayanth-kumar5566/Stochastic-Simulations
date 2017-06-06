import SIR_functions
import numpy
import  matplotlib.pyplot as plt
N1 = 1000
N2 = 1000
mu = 0
gamma = 0.01
omega = 0
tr12 = 0
tr21 = 0
tmax = 100
alpha = 0
beta1=0.008
beta2=0.008

#----------------------------------Averaging of the series-----------------------------------------------------------
'''
series=[]
time=[]
count=0
while count<=4:
    (t1ser,t2ser,tot,tim)=SIR_functions.st_sim(beta1,beta2,N1,N2,mu,gamma,omega,tr12,tr21,alpha)
    series.append(t1ser)
    time.append(tim)
    count += 1
avg=SIR_functions.avg_ser(series,time)
tim=avg[1]
ser=avg[0]
'''
#---------------------------------Fitting--------------------------------------------------------------------------------
'''
(t1ser,t2ser,tot,tim)=SIR_functions.st_sim(beta1,beta2,N1,N2,mu,gamma,omega,tr12,tr21,alpha)
ser=t1ser
tim=tim
y=SIR_functions.preprocessing(ser)
x=SIR_functions.finding_point(ser,tim,'max')
#x2=SIR_functions.finding_point(ser,tim,'slope')
#x=max(x1,x2)
plt.plot(tim[:x],numpy.log(ser[:x]),'b-',label='orginal series')
ser=ser[y:x]
time=tim[y:x]'''

'''s=SIR_functions.fit(ser,time)
plt.plot(time,SIR_functions.y_sl(time,s[0],s[1]),'r-',label='fit fn')
print s[0]
z=SIR_functions.Fitt(ser,time)
print z[0]
plt.plot(time,SIR_functions.y_sl(time,z[0],z[1]),'g-',label='Fitt fn')'''

'''
sl=SIR_functions.Rcode(tim,ser)
print sl
plt.plot(tim[y:x],SIR_functions.y_sl(tim[y:x],sl,0),'k-',label='R code')
plt.legend(loc='best')
plt.show()
'''

#----------------------------------------------------------------------------------------
#                            The Ro dependencies

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

