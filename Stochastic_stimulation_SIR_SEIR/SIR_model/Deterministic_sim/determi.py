import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def dydt2Cities(Y,t,beta1,beta2,gamma,tr12,tr21,lam,mu):
    S1,I1,R1,S2,I2,R2 = Y
    dS1dt = -beta1*S1*I1 -beta1*S1*tr21*I2 +tr21*S2 -tr12*S1 +lam*(S1+I1+R1) -mu*S1
    dI1dt = beta1*S1*I1 - gamma*I1 +tr21*I2*beta1*S1 -gamma*I2*tr21 +tr21*I2 -tr12*I1 -mu*I1
    dR1dt = gamma*I1 +gamma*tr21*I2+ tr21*R2 -tr12*R1 -mu*R1
    dS2dt = -beta2*S2*I2 -beta2*S2*tr12*I1 +tr12*S1 -tr21*S2 +lam*(S2+I2+R2) -mu*S2
    dI2dt = beta2*S2*I2 - gamma*I2 +tr12*I1*beta2*S2 -gamma*I1*tr12 +tr12*I1 -tr21*I2 -mu*I2
    dR2dt = gamma*I2 +gamma*tr12*I1+ tr12*R1 -tr21*R2 -mu*R2
    return np.array([dS1dt,dI1dt,dR1dt,dS2dt,dI2dt,dR2dt])

def Simulate(tmax,beta1,beta2,gamma,tr12,tr21,lam,mu):
 	Y_in = np.array([1000,1,0,1000,1,0])
	t = np.arange(0,tmax,0.01)
	Y = odeint(dydt2Cities,Y_in,t,args=(beta1,beta2,gamma,tr12,tr21,lam,mu,))
	#print "Ro is ", (beta1*1000)/float(gamma)	
	#fig,ax = plt.subplots()
	#ax.plot(t,Y,label=('S1','I1','R1','S2','I2','R2'))
	#ax.legend(loc='best')
	#plt.show()
	return (t,Y)
# if __name__ == '__main__':
#Simulate(1000,0.0002,0.0005,0.1,0.4,0.4)
tr=np.linspace(0,0.9,20)
labels=['S1','I1','R1','S2','I2','R2']

birth=0.0005
death=0.0002
'''
count=0
for i in tr:
	(t,Y)=Simulate(100,0.0002,0.0008,0.1,i,i,birth,death)
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

(t,Y)=Simulate(100,0.0002,0.0008,0.1,0,0,birth,death)
for y ,label in zip(Y.transpose(),labels):	
    plt.plot(t,y,label=label)
plt.legend(loc='best')
plt.show()

