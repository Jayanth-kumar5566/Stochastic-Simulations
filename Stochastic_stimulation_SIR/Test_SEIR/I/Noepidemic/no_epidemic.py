from __future__ import division
import matplotlib as ml
ml.use('agg')
import SEIR_functions
import SIR_functions
import numpy
import  matplotlib.pyplot as plt
from scipy import polyfit
import multiprocessing as mp
#----------------------------------Averaging of the series-----------------------------------------------------------
def sim_av(tr21,beta1=0.8,beta2=1.5):
    tr12=tr21
    series_t1=[]
    series_t2=[]
    time=[]
    count=0
    while count<=100:
        (t1ser,t2ser,tot,tim)=SEIR_functions.st_sim(beta1,beta2,lam,mu,N1,N2,gamma,sigma,tr12,tr21)
        series_t1.append(t1ser)
        series_t2.append(t2ser)
        time.append(tim)
        count += 1
    print "Averaging the iterations"
    #pool=mp.Pool()
    #(avg_t1,avg_t2)=pool.map(multi_run_wrapper,[(series_t1,time),(series_t2,time)])
    avg_t1=SEIR_functions.avg_ser(series_t1,time)
    avg_t2=SEIR_functions.avg_ser(series_t2,time)
    #avg_tot=SEIR_functions.avg_ser(series_total,time)
    tim_t1=avg_t1[1]
    ser_t1=avg_t1[0]
    tim_t2=avg_t2[1]
    ser_t2=avg_t2[0]
    plt.clf()
    plt.plot(tim_t1,ser_t1,'g',label='City1 infection')
    plt.plot(tim_t2,ser_t2,'r',label='City2 infection')
    plt.xlabel("Time")
    plt.ylabel("Number of people Infected")
    plt.title("SEIR"+"$\epsilon =$"+str(tr12))
    plt.legend(loc='best')
    plt.savefig('seir'+str(tr12)+'.png', format='png', orientation='landscape')
    plt.close()
    return [(ser_t1,tim_t1),(ser_t2,tim_t2)]
#--------------------------------------------------------------------------------------------------
def sim_av_1(tr21,beta1=0.8,beta2=1.5):
   tr12=tr21
   series_t1=[]
   series_t2=[]
   time=[]
   count=0
   while count<=100:
      (t1ser,t2ser,tot,tim)=SIR_functions.st_sim(beta1,beta2,N1,N2,0,gamma,0,tr12,tr21,0)
      series_t1.append(t1ser)
      series_t2.append(t2ser)
      time.append(tim)
      count += 1
   print "Averaging the iterations"
   #pool=mp.Pool()
   #(avg_t1,avg_t2,avg_tot)=pool.map(multi_run_wrapper,[(series_t1,time),(series_t2,time)])
   avg_t1=SEIR_functions.avg_ser(series_t1,time)
   avg_t2=SEIR_functions.avg_ser(series_t2,time)
   #avg_tot=SEIR_functions.avg_ser(series_total,time)
   tim_t1=avg_t1[1]
   ser_t1=avg_t1[0]
   tim_t2=avg_t2[1]
   ser_t2=avg_t2[0]
   #----If needed to check
   plt.clf()
   plt.plot(tim_t1,ser_t1,'g',label='series1')
   plt.plot(tim_t2,ser_t2,'r',label='series2')
   plt.legend(loc='best')
   plt.xlabel("Time")
   plt.ylabel("Number of people Infected")
   plt.title("SIR"+"$\epsilon =$"+str(tr12))
   plt.legend(loc='best')
   plt.savefig('sir'+str(tr12)+'.png', format='png', orientation='landscape')
   plt.close()
   return [(ser_t1,tim_t1),(ser_t2,tim_t2)]
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

N1 = 1000
N2 = 1000
mu = 0
lam =  0
gamma = 0.1
sigma = 20
tmax = 100
beta1=1.5
beta2=0.8
tr_val=numpy.linspace(0,0.7,20)
'''
for i in tr_val:
   p1=mp.Process(target=sim_av,args=(beta1,beta2,i,i))
   p2=mp.Process(target=sim_av_1,args=(beta1,beta2,i,i))
   p1.start()
   p2.start()
'''


mp.Pool(4).map(sim_av,tr_val) 
mp.Pool(4).map(sim_av_1,tr_val) 
