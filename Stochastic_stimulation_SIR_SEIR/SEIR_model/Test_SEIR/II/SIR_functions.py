#------------Functions that will be used--------------------------

#---------Importing modules-------------------------------------
from __future__ import division
import numpy
import random 
import matplotlib.pyplot as plt
import scipy.stats as ss
import pandas
import os
#-----------------Sctochastic simulations------------------------
def st_sim(beta1,beta2,N1,N2,lam,mu,gamma,omega,tr12,tr21,alpha):

    # Intial values:
    tmax = 100  #Default value can be changed if needed
    #print "Ro of city 1 =", beta1*N1/gamma
    #print "Ro of city 2 =", beta2*N1/gamma
    #print "expected slope",gamma*((beta1*N1/gamma)-1)
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
        Rate_S12I1 = (beta1*S1*I1)/(S1+I1+R1) +alpha*S1 
        Rate_I12R1 = gamma*I1 
        Rate_S22I2 = beta2*S2*I2/(S2+I2+R2) + alpha*S2
        Rate_I22R2 = gamma*I2
        Rate_S22S1 = tr21*S2
        Rate_I22I1 = tr21*I2
        Rate_R22R1 = tr21*R2
        Rate_S12S2 = tr12*S1
        Rate_I12I2 = tr12*I1
        Rate_R12R2 = tr12*R1
        Birth_1    = lam*(S1+I1+R1)
        Birth_2    = lam*(S2+I2+R2)
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
    ''' 
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
    '''    
    
    tot=I1Val+I2Val
    return (I1Val,I2Val,tot,TVal)
#--------------------Preprocessing----------------------------------------------
def preprocessing(ser):
    '''
    Input: Takes in a series
    output: Prints out the number of repeatations of the series'''
    y=numpy.where(numpy.diff(ser)>0)[0][0]+1
    return y
#-------------------Finding the point----------------------------------------
def finding_point(series,time,method='slope'):
    '''
    Input 
        Series: a numpy nD array
        time  : numpy nD array with same shape as series
        method: uses slope method as default
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
	a=0
        for i in range(len(slope)-1):
            if abs(slope[i]-slope[i+1])<=0.4 and abs(slope[i]-slope[i+1])!=0:
                a=i+1
                break
        return a
#---------------------------------Average of the series--------------------------------------
def avg_ser(series,time):
    '''---------Input---------
    series: Is a list of series(numpy nD array) that needs to be averaged the series must be split first
    time: Time values of the events as a list
    -------------Returns------------------
    avgeraged series: Numpy nD array'''

    no_o_ser=len(series)
    delta=numpy.mean(numpy.diff(time[0]))
    stop=min(map(len,series))
    x_intrp=numpy.arange(0,stop+delta,delta)
    y_intrp=numpy.zeros(len(x_intrp))
    for i,j in zip(series,time):
        y_intrp += numpy.interp(x_intrp,j,i)
    avg= y_intrp/no_o_ser
    return (avg,x_intrp)
#---------------------------------Fitting of the slope---------------------------------------
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
def Rcode(time,series):
    '''
    Input:
         time: The time series
         series: The series of the infection even before log, the series must be preprocessed using finding point with method max 
    Return
         The value of the slope and the intercept  as a float
    Note: The sigmoid_slope.R should be present in the same directory'''
    xdf=pandas.DataFrame(time)
    xdf.to_csv("x.csv")
    ydf=pandas.DataFrame(series)
    ydf.to_csv("y.csv")
    os.system("Rscript sigmoid_slope.R")
    file=open('tmp','r')
    a=file.readlines()
    a=a[0].strip('\n')
    (co,sl)=a.split(" ")
    os.system('rm x.csv')
    os.system('rm y.csv')
    os.system('rm tmp')
    return (float(co),float(sl))
def mv_Avg(ser,time,window):
    '''
    Input:
         ser: Takes in a series that is to be smoothed, after log is recommended
         time: Takes the respective time as of the points as the index
         window: Window over which they need to be smoothed
    output:
         A tuple with (infection_values,time of the series)
    '''
    ser=pandas.Series(data=ser,index=time)
    avg_ser=ser.rolling(window=window).mean()
    avg_ser_va=avg_ser.to_dict()
    return (avg_ser_va.values(),avg_ser_va.keys())
#-------------------------------------------------------------------------------------------
