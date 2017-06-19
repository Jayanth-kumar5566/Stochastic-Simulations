from __future__ import division
import SEIR_functions
import SIR_functions
import numpy
import  matplotlib.pyplot as plt
from scipy import polyfit
import sys
N1 = 1000
N2 = 1000
mu = 0
lam =  0
gamma = 0.1
sigma = 200
tmax = 100

#----------------------------------Averaging of the series-----------------------------------------------------------
def sim_av(beta1,beta2,tr12,tr21,output):
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
    output.put([(ser_tot,tim_tot)])
#--------------------------------------------------------------------------------------------------
def sim_av_1(beta1,beta2,tr12,tr21,output):
    #series_t1=[]
    #series_t2=[]
    series_total=[]
    time=[]
    count=0
    num_iter=0
    while count<=50:
        (t1ser,t2ser,tot,tim)=SIR_functions.st_sim(beta1,beta2,N1,N2,0,gamma,0,tr12,tr21,0)
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
    output.put([(ser_tot,tim_tot)])

#---------------------------------Fitting--------------------------------------------------------------------------------
def fit((ser,tim)):
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
    return sl[1]
#-------------------------------------------------------------------------------------------

beta1=1.5
beta2=0.8
tr_val=numpy.linspace(1e-7,0.7,50)

import multiprocessing as mp
count=0
seir=[]
sir=[]
for (n,m) in zip(tr_val,tr_val):
    print "Transfer value and simulating", n
    output=mp.Queue()
    output1=mp.Queue()
    p1=mp.Process(target=sim_av,args=(beta1,beta2,n,n,output))
    p1.start()
    p2=mp.Process(target=sim_av_1,args=(beta1,beta2,n,n,output1))
    p2.start()
    [(seir_tot,seir_tim_tot)]=output.get()
    [(sir_tot,sir_tim_tot)]=output1.get()
    print "Fitting"
    (zz,yy)=map(fit,[(seir_tot,seir_tim_tot),(sir_tot,sir_tim_tot)])
    seir.append(1+(zz/gamma))
    sir.append(1+(yy/gamma))
plt.clf()
plt.figure(figsize=(30,15))
plt.plot(tr_val,sir,'bo',label='SIR totalR0')
plt.plot(tr_val,seir,'ro',label='Total R0 SEIR')
plt.xlabel("Transfer rate")
plt.ylabel("$R_{0}$ Values")
plt.legend(loc='best')
plt.savefig('III_200_high_res.png', format='png', orientation='landscape')
plt.close()
