import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def dydt2Cities(Y,t,beta1,beta2,gamma):
    S1,I1,R1,S2,I2,R2 = Y
    dS1dt = -beta1*S1*I1
    dI1dt = beta1*S1*I1 - gamma*I1
    dR1dt = gamma*I1
    dS2dt = -beta2*S2*I2
    dI2dt = beta2*S2*I2 - gamma*I2
    dR2dt = gamma*I2
    return np.array([dS1dt,dI1dt,dR1dt,dS2dt,dI2dt,dR2dt])

def Simulate(tmax,beta1,beta2,gamma):
    Y_in = np.array([1000,1,0,1000,1,0])
    t = np.arange(0,tmax,0.001)
    Y = odeint(dydt2Cities,Y_in,t,args=(beta1,beta2,gamma,))
    fig,ax = plt.subplots()
    ax.plot(t,Y)
    plt.show()

# if __name__ == '__main__':
Simulate(100,0.0009,0.0009,0.1)
     

    
