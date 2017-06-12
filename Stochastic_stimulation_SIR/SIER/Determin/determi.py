from __future__ import division
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def dydt2Cities(Y,t,beta1,gamma,alpha):
    S1 = Y[0]
    E1 = Y[1]
    I1 = Y[2]
    R1 = Y[3]
    N=S1+E1+I1+R1
    print S1
    dS1dt = -(beta1*S1*I1)/N
    dE1dt =  (beta1*S1*I1)/N -alpha*E1 
    dI1dt =  alpha*E1 - gamma*I1 
    dR1dt =  gamma*I1 
    return np.array([dS1dt,dI1dt,dR1dt])

def Simulate(tmax,beta1,gamma,alpha):
 	Y_in = np.array([10,1,1,0])
	t = np.arange(0,tmax,0.001)
	Y = odeint(dydt2Cities,Y_in,t,args=(beta1,gamma,alpha))
	#print "Ro is ", (beta1*1000)/float(gamma)	
        '''
        fig,ax = plt.subplots()
	ax.plot(t,Y)
	plt.show()
	'''
        return (t,Y)
#Simulate(1000,1,1,1)    



labels=['S1','E1','I1','R1']
(t,Y)=Simulate(15,0.9,0.2,0.5)
for y ,label in zip(Y.transpose(),labels):	
    plt.plot(t,y,label=label)
plt.legend(loc='best')
plt.show()





# if __name__ == '__main__':
#Simulate(1000,0.0002,0.0005,0.1,0.4,0.4)

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


