from __future__ import division
import SEIR_functions as sfunc
#import matplotlib.pyplot as plt
import numpy
beta1=0.8
beta2=1.5
lam=1e-7
mu=1e-7
N1=1000
N2=1000
gamma=0.1
sigma=1
#e=0
#------ -------------------------------------------------------------------------------
tr_val=[]
total_ro=[]
std_ro=[]

for e in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
    count=0
    r=[]
    while count<500:
        avg_count=0
        total_series=[]
        time_ser=[]
        while avg_count<50:
            (t1ser,t2ser,totser,time)=sfunc.st_sim(beta1,beta2,lam,mu,N1,N2,gamma,sigma,e,e)
            if numpy.max(t1ser) > 0.1*N1 or numpy.max(t2ser) > 0.1*N2:
                total_series.append(totser)
                time_ser.append(time)
                avg_count += 1
                '''
                plt.plot(time,t1ser)
                plt.plot(time,t2ser)
                plt.show()
                '''
        (ser,time)=sfunc.avg_ser(total_series,time_ser)
        y=sfunc.preprocessing(ser)
        x=sfunc.finding_point(ser,time,method='max')
        '''
        plt.plot(time[:x],numpy.log(ser[:x]),'b-',label='orginal series')
        '''
        ser=ser[y:x]
        t=time[y:x]
        try:
            (intercept,slope,rsq)=sfunc.Rcode(t,ser)
        except:
            print "Error in fitting, passing on"
            rsq=0
        if rsq>0.75:
            r.append(slope)

            '''
            plt.plot(t,sfunc.y_sl(t,slope,intercept),'k-',label='R code')
            plt.legend(loc='best')
            plt.show()
            '''
            print count
            count=count+1

    total_ro.append(numpy.mean(r))
    std_ro.append(numpy.std(r))
    tr_val.append(e)
    print (numpy.mean(r),numpy.std(r))

 
'''
plt.errorbar(tr_val, total_ro, yerr=std_ro, fmt='o')
plt.plot(tr_val,[0.612970334961301]*len(tr_val),'-')
plt.show()
'''
file=open("values.csv",'w')
file.write("tr_val,total_r0,std\n")
for i in range(len(tr_val)):
    file.write(str(tr_val[i])+','+str(total_ro[i])+','+str(std_ro[i])+'\n')

file.close()
