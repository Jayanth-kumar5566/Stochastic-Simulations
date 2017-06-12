from __future__ import division
import numpy as np
from scipy.integrate import odeint
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


def dydt2Cities(Y,t,beta1,beta2,gamma,sigma,tr12,tr21):
    S1 = Y[0]
    E1 = Y[1]
    I1 = Y[2]
    R1 = Y[3]
    S2 = Y[4]
    E2 = Y[5]
    I2 = Y[6]
    R2 = Y[7]
    dS1dt = -beta1*S1*I1 -tr12*S1 +tr21*S2
    dE1dt =  beta1*S1*I1 -sigma*E1 -tr12*E1 +tr21*E2
    dI1dt =  sigma*E1 - gamma*I1 
    dR1dt =  gamma*I1 -tr12*R1 +tr21*R2
    dS2dt = -beta2*S2*I2 -tr21*S2 +tr12*S1
    dE2dt =  beta2*S2*I2 -sigma*E2 -tr21*E2 +tr12*E1
    dI2dt =  sigma*E2 - gamma*I2
    dR2dt =  gamma*I2 -tr21*R2 +tr12*R1
    return np.array([dS1dt,dE1dt,dI1dt,dR1dt,dS2dt,dE2dt,dI2dt,dR2dt])
 
def Simulate(tmax,beta1,beta2,gamma,sigma,tr12,tr21):
 	Y_in = np.array([1000,1,0,0,1000,0,0,0])
	t = np.arange(0,tmax,0.001)
	Y = odeint(dydt2Cities,Y_in,t,args=(beta1,beta2,gamma,sigma,tr12,tr21))
	#print "Ro is ", (beta1*1000)/float(gamma)	
        '''
        fig,ax = plt.subplots()
	ax.plot(t,Y)
	plt.show()
	'''
        return (t,Y)
    


fig,ax=plt.subplots(2,sharex=True)
(t,Y)=Simulate(100,0.09,0.09,0.2,0.5,0.1,0.1)
Y=Y.transpose()
ax[0].plot(t,Y[0],label="S1")
ax[0].plot(t,Y[1],label="E1")
ax[0].plot(t,Y[2],label="I1")
ax[0].plot(t,Y[3],label="R1")
ax[1].plot(t,Y[4],label="S2")
ax[1].plot(t,Y[5],label="E2")
ax[1].plot(t,Y[6],label="I2")
ax[1].plot(t,Y[7],label="R2")
ax[1].legend(loc='best')
ax[0].legend(loc='best')
plt.savefig("test.pdf")






'''
count=0
for i in tr:
	(t,Y)=Simulate(100,0.0002,0.0008,0.1,i,i)
	plt.clf()	
	plt.figure(figsize=(30,15))	
	for y ,label in zip(Y.transpose(),labels):	
		plt.plot(t,y,label=label)
	plt.legend(loc='best')
	plt.savefig(str(count)+'.png', format='png', orientation='landscape')
	count += 1
	plt.close()
'''
'''
file=open('y.csv','w')
for i in z:
	file.write(str(i)+'\n')
file.close()
'''


