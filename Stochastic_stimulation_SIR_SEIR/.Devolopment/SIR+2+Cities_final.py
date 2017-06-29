
# coding: utf-8

# # SIR model on 2 Cities with population Migration

# ## Parameters

# - ### City 1
#     -  S_1 Susceptible Individuals
#     -  I_1 Infected Individuals
#     -  R_1 Recovered Individuals
#     -  $N_{1}$ Total Population of City 1
# - ### City 2
#     -  S_2 Susceptible Individuals
#     -  I_2 Infected Individuals
#     -  R_2 Recovered Individuals
#     -  $N_{2}$ Total Population of City 2
# - $\beta_{1}$ Transmission Rate of Infection in City 1
# - $\beta_{2}$ Transmission Rate of Infection in City 2
# - $\beta$ can be different since they can depend upon the network of people inside the cities
# - $\gamma_{1}$ Recovery rate in City 1
# - $\gamma_{2}$ Recovery rate in City 2
# - Note: $\gamma_{1}$ and $\gamma_{2}$ cannot be different for different cities as it depends on the disease
# - $\mu$ Rate of Birth and Death
# - $\omega$ Pathogen induced mortality rate
# - $\zeta_{1,2}$ Migration rate from  $1 \rightarrow 2$
# - $\zeta_{2,1}$ Migration rate from $2 \rightarrow 1$
# - $\alpha$ The Probability that the infection comes from outside the system

# ## Assumptions

# - We assume that if a person migrates then, he/she doesnot undergo death or give birth

# ## Equations 
# ### City 1

# #### Susceptible 
# $$\frac{dS_{1}}{dt}=\mu( N_{1} - S_{1}) - \beta_{1} S_{1}(I_{1}+\zeta_{2,1} I_{2}) + \zeta_{2,1} S_{2} -\zeta_{1,2} S_{1} - \alpha S_{1}$$

# #### Infected
# $$\frac{dI_{1}}{dt}=\alpha S_{1}+ \beta_{1} S_{1}(I_{1}+\zeta_{2,1}I_{2}) -(\mu+\omega)I_{1} - \gamma I_{1} + \zeta_{2,1}I_{2} - \zeta_{1,2} I_{1}$$

# #### Recovered
# $$\frac{dR_{1}}{dt}= \gamma I_{1} -\mu R_{1} + \zeta_{2,1} R_{2} - \zeta_{1,2} R_{1} $$

# ### City 2

# #### Susceptible 
# $$\frac{dS_{2}}{dt}=\mu( N_{2} - S_{2}) - \beta_{2} S_{2}(I_{2}+\zeta_{1,2} I_{1}) + \zeta_{1,2} S_{1} -\zeta_{2,1} S_{2} - \alpha S_{2}$$

# #### Infected
# $$\frac{dI_{2}}{dt}= \alpha S_{2}+ \beta_{2} S_{2}(I_{2}+\zeta_{1,2}I_{1}) -(\mu+\omega)I_{2} - \gamma I_{2} + \zeta_{1,2}I_{1} - \zeta_{2,1} I_{2}$$

# #### Recovered
# $$\frac{dR_{2}}{dt}= \gamma I_{2} -\mu R_{2} + \zeta_{1,2} R_{1} -\zeta_{2,1} R_{2} $$

# ### Constraint Equations (Does not matter for Stochastic Model)

# $$N_{1}=S_{1}+I_{1}+R_{1},N_{2}=S_{2}+I_{2}+R_{2}$$

# $$\frac{dN_{1}}{dt}+\frac{dN_{2}}{dt}= 0$$

# $$\frac{dN_{1}}{dt}=\zeta_{2,1} N_{2} - \zeta_{1,2} N_{1}$$

# <span style="color:Blue">To do:</span> The Constraint equation is not satisfied $-\omega(I_{1}+I_{2})$ term remains.
# <span style="color:Blue">Ans:</span> The constraint equation are not supposed to be satisfied since the model is stochastic 





















#=======================================================Simulations==================================================
# # Simulating the above the Model

# Importing the Necessary modules:

from __future__ import division
import numpy
import random 
import matplotlib.pyplot as plt
from pandas import Series
import scipy.stats as ss

# ## Fill up the parameters required for the simulation 


def st_sim(beta1,beta2):

    # Intial values:

    N1 = 1000
    N2 = 1000
    mu = 0
    #beta1 = 2
    #beta2 = 2
    gamma1 = 0.1
    gamma2 = gamma1
    omega = 0
    tr12 = 0
    tr21 = 0
    tmax = 1000
    alpha = 0
    print "Ro of city 1 =", beta1*N1/gamma1
    print "Ro of city 2 =", beta2*N1/gamma1
    print "expected slope",gamma1*((beta1*N1/gamma1)-1)
    # In[12]:

    MAX=int(1e6)
    TVal=numpy.zeros(MAX,dtype=float)
    S1Val=numpy.zeros(MAX,dtype=int)
    I1Val=numpy.zeros(MAX,dtype=int)
    R1Val=numpy.zeros(MAX,dtype=int)
    S2Val=numpy.zeros(MAX,dtype=int)
    I2Val=numpy.zeros(MAX,dtype=int)
    R2Val=numpy.zeros(MAX,dtype=int)


    #------------Initial Values-----------------------
    count = 0
    t = 0

    I1    = 1
    R1    = 0
    S1    = N1-I1
    I2    = 1
    R2    = 0
    S2    = N2

    TVal[count]=t
    S1Val[count]=S1
    I1Val[count]=I1
    R1Val[count]=R1
    S2Val[count]=S2
    I2Val[count]=I2
    R2Val[count]=R2

    while count < MAX and t < tmax and I1>0 and I2>0:
        Rate_S12I1 = beta1*S1*(I1+tr21*I2)+alpha*S1 
        Rate_I12R1 = gamma1*(I1+tr21*I2) 
        Rate_S22I2 = beta2*S2*(I2+tr12*I1)+alpha*S2
        Rate_I22R2 = gamma2*(I1+tr12*I1)
        Rate_S22S1 = tr21*S2
        Rate_I22I1 = tr21*I2
        Rate_R22R1 = tr21*R2
        Rate_S12S2 = tr12*S1
        Rate_I12I2 = tr12*I1
        Rate_R12R2 = tr12*R1
        Birth_1    = mu*(S1+I1+R1)
        Birth_2    = mu*(S2+I2+R2)
        Death_S1   = mu*S1
        Death_S2   = mu*S2
        Death_I1   = (mu+omega)*I1
        Death_I2   = (mu+omega)*I2
        Death_R1   = mu*R1
        Death_R2   = mu*R2

        K  = Rate_S12I1+Rate_I12R1+Rate_S22I2+Rate_I22R2+Rate_S22S1+Rate_I22I1+Rate_R22R1+Rate_S12S2+Rate_I12I2+Rate_R12R2+Birth_1+Birth_2+Death_S1+Death_S2+Death_I1+Death_I2+Death_R1+Death_R2

        dt = -(1.0/K)*numpy.log(random.random())

        t = t + dt
        count = count + 1

        r= random.random()*K

        if r < Rate_S12I1:
            S1 -= 1
            I1 += 1
        elif r < Rate_S12I1+Rate_I12R1:
            I1 -= 1
            R1 += 1
        elif r < Rate_S12I1+Rate_I12R1+Rate_S22I2:
            S2 -= 1
            I2 += 1
        elif r < Rate_S12I1+Rate_I12R1+Rate_S22I2+Rate_I22R2:
            I2 -= 1
            R2 += 1
        elif r < Rate_S12I1+Rate_I12R1+Rate_S22I2+Rate_I22R2+Rate_S22S1:
            S2 -= 1
            S1 += 1
        elif r < Rate_S12I1+Rate_I12R1+Rate_S22I2+Rate_I22R2+Rate_S22S1+Rate_I22I1:
            I2 -= 1
            I1 += 1
        elif r < Rate_S12I1+Rate_I12R1+Rate_S22I2+Rate_I22R2+Rate_S22S1+Rate_I22I1+Rate_R22R1:
            R2 -= 1
            R1 += 1
        elif r < Rate_S12I1+Rate_I12R1+Rate_S22I2+Rate_I22R2+Rate_S22S1+Rate_I22I1+Rate_R22R1+Rate_S12S2:
            S1 -= 1
            S2 += 1
        elif r < Rate_S12I1+Rate_I12R1+Rate_S22I2+Rate_I22R2+Rate_S22S1+Rate_I22I1+Rate_R22R1+Rate_S12S2+Rate_I12I2:
            I1 -= 1
            I2 += 1
        elif r < Rate_S12I1+Rate_I12R1+Rate_S22I2+Rate_I22R2+Rate_S22S1+Rate_I22I1+Rate_R22R1+Rate_S12S2+Rate_I12I2+Rate_R12R2:
            R1 -= 1
            R2 += 1    
        elif r < Rate_S12I1+Rate_I12R1+Rate_S22I2+Rate_I22R2+Rate_S22S1+Rate_I22I1+Rate_R22R1+Rate_S12S2+Rate_I12I2+Rate_R12R2+Birth_1:
            S1 += 1
        elif r < Rate_S12I1+Rate_I12R1+Rate_S22I2+Rate_I22R2+Rate_S22S1+Rate_I22I1+Rate_R22R1+Rate_S12S2+Rate_I12I2+Rate_R12R2+Birth_1+Birth_2:
            S2 +=1
        elif r < Rate_S12I1+Rate_I12R1+Rate_S22I2+Rate_I22R2+Rate_S22S1+Rate_I22I1+Rate_R22R1+Rate_S12S2+Rate_I12I2+Rate_R12R2+Birth_1+Birth_2+Death_S1:
            S1 -=1
        elif r < Rate_S12I1+Rate_I12R1+Rate_S22I2+Rate_I22R2+Rate_S22S1+Rate_I22I1+Rate_R22R1+Rate_S12S2+Rate_I12I2+Rate_R12R2+Birth_1+Birth_2+Death_S1+Death_S2:
            S2 -=1
        elif r < Rate_S12I1+Rate_I12R1+Rate_S22I2+Rate_I22R2+Rate_S22S1+Rate_I22I1+Rate_R22R1+Rate_S12S2+Rate_I12I2+Rate_R12R2+Birth_1+Birth_2+Death_S1+Death_S2+Death_I1:
            I1 -=1
        elif r < Rate_S12I1+Rate_I12R1+Rate_S22I2+Rate_I22R2+Rate_S22S1+Rate_I22I1+Rate_R22R1+Rate_S12S2+Rate_I12I2+Rate_R12R2+Birth_1+Birth_2+Death_S1+Death_S2+Death_I1+Death_I2:
            I2 -=1
        elif r < Rate_S12I1+Rate_I12R1+Rate_S22I2+Rate_I22R2+Rate_S22S1+Rate_I22I1+Rate_R22R1+Rate_S12S2+Rate_I12I2+Rate_R12R2+Birth_1+Birth_2+Death_S1+Death_S2+Death_I1+Death_I2+Death_R1:
            R1 -=1
        elif r < Rate_S12I1+Rate_I12R1+Rate_S22I2+Rate_I22R2+Rate_S22S1+Rate_I22I1+Rate_R22R1+Rate_S12S2+Rate_I12I2+Rate_R12R2+Birth_1+Birth_2+Death_S1+Death_S2+Death_I1+Death_I2+Death_R1+Death_R2:
            R2 -=1
        TVal[count]=t
        S1Val[count]=S1
        I1Val[count]=I1
        R1Val[count]=R1
        S2Val[count]=S2
        I2Val[count]=I2
        R2Val[count]=R2
    TVal=TVal[:count+1]
    S1Val=S1Val[:count+1]
    I1Val=I1Val[:count+1]
    R1Val=R1Val[:count+1]
    S2Val=S2Val[:count+1]
    I2Val=I2Val[:count+1]
    R2Val=R2Val[:count+1]

    #print 'Number of events = ',count
    #-------------------------------------------#
    
    fig,ax = plt.subplots(2,sharex=True)
    ax[0].plot(TVal,S1Val,'b-',label='S1')
    ax[0].plot(TVal,I1Val,'r-',label='I1')
    ax[0].plot(TVal,R1Val,'g-',label='R1')
    ax[1].plot(TVal,S2Val,'b-',label='S2')
    ax[1].plot(TVal,I2Val,'r-',label='I2')
    ax[1].plot(TVal,R2Val,'g-',label='R2')
    ax[1].set_xlabel('time')
    ax[1].legend(loc='best')
    plt.show()    
    
    tot=I1Val+I2Val
    return (I1Val,I2Val,tot,TVal)

'''
fig,ax = plt.subplots()
ax.plot(numpy.log(t1ser),label='T1')
ax.plot(numpy.log(t2ser),label='T2')
ax.plot(numpy.log(tot),label='Total')
ax.legend(loc='best')
plt.show()'''

#=----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------Preprocessing----------------------------------------
def preprocessing(ser):
    '''
    if 0 in numpy.diff(ser):
        x=numpy.diff(ser)
        c=1
        for i in x:
            if i == 0:
                c += 1
            else:
                break
        return c
    else:
        return 0'''
    y=numpy.where(numpy.diff(ser)>0)[0][0]+1
    return y
#--------------------------------------------------------------------------------------------------------------
def finding_point(series,time,method='max'):
    '''
    Input 
        Series: a numpy nD array
        time  : numpy nD array with same shape as series
    Returns
       (start,stop): Indices for the start and stop for the series'''

    if 0 in series:
        print "Zero values present in the series, Please use preprocesing to input the series. If already used please ignore"

    ser=numpy.log(series)
    if method == "max":
        b=numpy.nanargmax(ser)
        return int(b/2)
    else:
        slop_num=numpy.diff(ser)
        slop_den=numpy.diff(time)
        slope=slop_num/slop_den
        for i in range(len(slope)-1):
            if abs(slope[i]-slope[i+1])<=0.4 and abs(slope[i]-slope[i+1])!=0:
                a=i+1
                break
        return a

'''
def Derivative(TSER,T):
    Diff = numpy.diff(TSER)
    idx = numpy.nonzero(Diff)[0]
    TSER = TSER[idx]
    Diff = numpy.diff(TSER)
    T    = T[idx]
    fig,ax  = plt.subplots()
    ax.plot(T[1:], Diff/numpy.diff(T),'-')
    plt.show() 
''' 

#---------------------Checking the split------------------------------
(t1ser,t2ser,tot,tim)=st_sim(0.00015,0.00015)
#Derivative(t1ser,tim)
x=finding_point(t1ser,tim,'max')
y=preprocessing(t1ser)
plt.plot(tim,t1ser,'b-')
plt.plot(tim[y:x],t1ser[y:x],'go')
plt.yscale('log')
plt.show()

#------------------------------------------------------------------------------------
def avg_ser(series,time):
    '''---------Input---------
    series: Is a list of series(numpy nD array) that needs to be averaged the series must be split first
    time: Time values of the events as a list
    -------------Returns------------------
    avgeraged series: Numpy nD array'''

    no_o_ser=len(series)
    delt=[]
    for i in time:
        delt.append(numpy.mean(numpy.diff(i)))
    delta=min(delt)
    stop=min(map(len,series))
    x_intrp=numpy.arange(0,stop+delta,delta)
    y_intrp=numpy.zeros(len(x_intrp))
    for i,j in zip(series,time):
        y_intrp += numpy.interp(x_intrp,j,i)
    avg= y_intrp/no_o_ser
    return (avg,x_intrp)
#-----------Checking Averge series--------------------------
'''
series=[]
time=[]
count=0
while count<=4:
    (t1ser,t2ser,tot,tim)=st_sim(0.8,0.6)
    series.append(t1ser)
    time.append(tim)
    plt.plot(tim,t1ser,label='series'+str(count))
    count += 1
avg=avg_ser(series,time)
plt.plot(avg[1],avg[0],label='avg series')
plt.legend(loc='best')
plt.show()
'''
#-------------------------------------------------------------------------------

#----------------Fitting the slope---------------------------------------------
def fit(series,time):
    '''
    Input:
         series: Is a numpy nD array is converted into log
         time  : Is a numpy nD array same shape as series
    Method: Uses correlation coefficient from the maxvalue and keeps on reducing the values 
    Returns:
         slope: The value of slope based on linear regression'''

    cor=[0,1]
    p=[0,1]
    
    x=cor[-1]
    y=cor[-2]

    series=numpy.log(series)
    ind=len(series)

    while abs(x-y) < 0.01:
        (p_r,p_p)=ss.pearsonr(time[:ind],series[:ind])
        numpy.append(cor,p_r)
        numpy.append(p,p_p)
        ind -= 1

    time=time[:ind]
    series=series[:ind]
    A=numpy.vstack([time,numpy.ones(len(time))]).T
    slop,lamb=numpy.linalg.lstsq(A,series)[0]
    return (slop,lamb)

def y_sl(x,slope,lamb):
    return slope*x+lamb

def Fitt(series,time):
    series=numpy.log(series)
    x=0
    parts=round(len(series)/10)
    cor=[]
    p  =[]
    slop=[]
    inte=[]
    while x+parts<=len(series):
        ser=series[int(x):int(x+parts)]
        tim=time[int(x):int(x+parts)]
        x += 1
        A=numpy.vstack([tim,numpy.ones(len(tim))]).T
        slope,inrp=numpy.linalg.lstsq(A,ser)[0]
        slop.append(slope)
        inte.append(inrp)
        (p_r,p_p)=ss.pearsonr(tim,ser)
        cor.append(p_r)
        p.append(p_p)
    p=numpy.array(p)
    cor=numpy.array(cor)
    p_new=1/p
    p_nor=(p_new-numpy.nanmin(p_new))/(numpy.nanmax(p_new)-numpy.nanmin(p_new))
    cor_nor=(cor-numpy.nanmin(cor))/(numpy.nanmax(cor)-numpy.nanmin(cor))
    new_metr=cor_nor+p_nor
    ind=numpy.nanargmax(new_metr)
    return (slop[ind],inte[ind])
        
#------------------------Checking the fit code--------------------------------------
'''
series=[]
time=[]
count=0
while count<=4:
    (t1ser,t2ser,tot,tim)=st_sim(0.02,0.02)
    series.append(t1ser)
    time.append(tim)
    count += 1
avg=avg_ser(series,time)
tim=avg[1]
ser=avg[0]
y=preprocessing(ser)
x=finding_point(ser,tim,'max')
plt.plot(tim[:x],numpy.log(ser[:x]),'b-',label='orginal series')
ser=ser[y:x]
time=tim[y:x]
z=Fitt(ser,time)
print z[0]
plt.plot(time,y_sl(time,z[0],z[1]),'g-',label='Fitt fn')
plt.legend(loc='best')
plt.show()
'''

'''
(t1ser,t2ser,tot,tim)=st_sim(0.02,0.02)
ser=t1ser
tim=tim
y=preprocessing(ser)
x=finding_point(ser,tim,'max')
plt.plot(tim[:x],numpy.log(ser[:x]),'b-',label='orginal series')
ser=ser[y:x]
time=tim[y:x]
s=fit(ser,time)
plt.plot(time,y_sl(time,s[0],s[1]),'r-',label='fit fn')
print s[0]
z=Fitt(ser,time)
print z[0]
plt.plot(time,y_sl(time,z[0],z[1]),'g-',label='Fitt fn')
#-------------------------To use R code--------------------------------------------

import pandas
#(t1ser,t2ser,tot,tim)=st_sim(0.8,0.6)
xdf=pandas.DataFrame(tim[y:])
xdf.to_csv("x.csv")
ydf=pandas.DataFrame(t1ser[y:])
ydf.to_csv("y.csv")
import os
os.system("Rscript sigmoid_slope.R")
file=open('tmp','r')
a=file.readlines()
sl=float(a[0].strip('\n'))
print sl
plt.plot(time,y_sl(time,sl,0),'k-',label='R code')
plt.legend(loc='best')
plt.show()
'''
#--------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------
#                            The Ro dependencies
'''
beta_values=[(0.01,0.8),(0.02,0.2),(0.04,0.9),(1,0.5)]
ro=[]
for (i,j) in beta_values:
    (t1ser,t2ser,tot,tim)=st_sim(i,j)
    r=[]
    for k in [t1ser,t2ser,tot]:
        y=preprocessing(k)
        x=finding_point(k,tim,'max')
        x=int(x/2)
        k_ser=k[y:x]
        time=tim[y:x]
        s=fit(k_ser,time)
        ''''''
        if s[0] <0:
            plt.plot(time,k_ser)
            def y(x,slope,lamb):
                return slope*x+lamb
            plt.plot(time,y(time,s[0],s[1]))
            plt.show()
        ''''''
        r.append(s[0])
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
'''

#-----------------------------------------r0 value dependeces----------------------------------------
'''
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
'''
#-----------------------------------------------------------------------------------------------------------












































#==============================================Version 1===========================================================
#---------------------------------------------------------------------------------------
'''
# In[14]:

import r0


# In[15]:

#Averages over all windows 2,3,4,5
def r_o_t(series,gamma2):
    r_x=[]
    #c_c=[]
    for i in [2,3,4,5]: #window moving average sizes
        (r,cor,p)=r0.r0(series,i,gamma2)
        #fig,ax = plt.subplots(1,2,sharey=True)
        #ax[0].plot(cor,r,'-o',label=str(i))
        #ax[1].plot(p,r,'-o',label=str(i))
        #ax[0].set_ylabel("Ro value")
        #ax[0].set_xlabel("Correlation Coefficient")
        #ax[1].set_xlabel("P value")
        #------------Choosing the appropriate r value----------------------
        p=numpy.array(p)
        cor=numpy.array(cor)
        p_new=1/p
        p_nor=(p_new-numpy.nanmin(p_new))/(numpy.nanmax(p_new)-numpy.nanmin(p_new))
        cor_nor=(cor-numpy.nanmin(cor))/(numpy.nanmax(cor)-numpy.nanmin(cor))
        new_metr=cor_nor+p_nor
        ind=numpy.nanargmax(new_metr)
        r_x.append(r[ind])
        #ax[1].legend(loc='best')
    #plt.plot()
    #print r_x
    return numpy.nanmean(r_x)


# In[25]:

def Mean(z):
    return (z[0]+z[1])/2
def Max(z):
    return max(z[0],z[1])
def Min(z):
    return min(z[0],z[1])
def X(z):
    return abs(z[0]-z[1])

# In[26]:

me=[]
ma=[]
mi=[]
Xax=[]
y=[]

# In[16]:

#-----------------------------------------------------Averaging over a lot of simulations---------------------------------------------
for beta2 in [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
    beta1=0.2
    ss=st_sim(beta1,beta2)
    t1ser=Series(data=ss[0])
    t2ser=Series(data=ss[1])
    tot=Series(data=ss[2])
    count = 0
    while count <= 10:
        z=st_sim(beta1,beta2)
        t1ser += Series(data=z[0])
        t2ser += Series(data=z[1])
        tot   += Series(data=z[2])
        count += 1
    
    #----------To check the averaging of the series-----------------------
    plt.plot(t1ser)
    plt.plot(t2ser)
    plt.plot(tot)
    plt.show()
    #----------------------------------------------------------------------------
    
    gamma2=0.02
    z=(r_o_t(t1ser,gamma2),r_o_t(t2ser,gamma2),r_o_t(tot,gamma2))
    #r_o_t(tot)
    print z

    me.append(Mean(z))
    ma.append(Max(z))
    mi.append(Min(z))
    Xax.append(X(z))
    y.append(z[2])
    
# In[147]:

plt.plot(Xax,y,'bo',label='actual total0')
plt.plot(Xax,me,'go',label='mean of ro')
plt.plot(Xax,ma,'ro',label='max of the ro')
plt.plot(Xax,mi,'ko',label='min of the ro')
plt.legend(loc='best')
plt.show()

'''

'''
# # Calculating $R_{0}$ more into version(Not to run, unless you know what u are doing)

# In[28]:

file=open("series1.csv",'w')
for i in t1ser:
    file.write(str(i)+',')
file.close()


# In[ ]:

file=open("series2.csv",'w')
for i in t2ser:
    file.write(str(i)+',')
file.close()


# In[ ]:

file=open("seriestot.csv",'w')
for i in gh:
    file.write(str(i)+',')
file.close()


# # Calculating the Error in the Estimation of $R_{0}$ using bootstraping

# In[ ]:

count=0
b=[]
while count<100:
    series=numpy.random.choice(t1ser[:60],len(t1ser[:60]),replace=True)
    mean=[]
    for i in range(15,50):
        mean.append(r_o_t(i,series))
    b.append(numpy.nanmean(mean))
    count += 1
    #print count
print numpy.nanmean(b)


# In[ ]:

b=numpy.array(b)
plt.hist(b[~numpy.isnan(b)])


'''
