from __future__ import division
import pandas
import matplotlib.pyplot as plt
df=pandas.read_csv("values.csv")
gamma=0.1
sigma=1
x=df['tr_val']
y=(1+(df['total_r0']/gamma))*(1+(df['total_r0']/sigma))
yerr=2*(df['std'])
plt.ylim(7,16)
plt.xlabel('$\epsilon$',fontsize=15)
plt.ylabel('Effective $R_{0}$',fontsize=15)
plt.title("Infection starting at both the settlements at same time\n"+'$\gamma=0.1$,'+r'$\beta_{1}=0.8$, $\beta_{2}=1.5$ ,$\sigma=1$'+',$N1=N2=1000$,$\lambda=\mu=10^{-7}$',fontsize=20)
plt.plot(x,[11.499987350012765]*len(x),'-',label='Average $R_{0}$')
plt.plot(x,[14.999983500016649]*len(x),'-',label='$R_{0}$ of Settlement 2')
plt.text(0.7,14.8,'$R_{0}$ of settlement2',fontsize=15)
plt.text(0.7,7.8,'$R_{0}$ of settlement1',fontsize=15)
plt.text(0.7,11.3,'Average of $R_{0}$ of settlements',fontsize=15)
plt.plot(x,[7.999991200008879]*len(x),'-',label='$R_{0}$ of Settlement 1')
plt.errorbar(x,y,yerr=yerr,fmt='o',color='k')
#plt.legend()
plt.show()
