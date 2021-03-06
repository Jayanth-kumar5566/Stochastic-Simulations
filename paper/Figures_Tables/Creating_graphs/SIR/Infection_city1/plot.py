import pandas
import matplotlib.pyplot as plt
df=pandas.read_csv("values.csv")
gamma=0.1
x=df['tr_val']
y=1+(df['total_r0']/gamma)
yerr=df['std']
plt.ylim(7,16)
plt.xlabel('$\epsilon$',fontsize=15)
plt.ylabel('Effective $R_{0}$',fontsize=15)
plt.title("Infection starting at Settlement 1\n"+'$\gamma=0.1$,'+r'$\beta_{1}=0.8$, $\beta_{2}=1.5$'+',$N1=N2=1000$,$\lambda=\mu=10^{-7}$',fontsize=20)
plt.plot(x,[11.5]*len(x),'-',label='Average $R_{0}$')
plt.plot(x,[15]*len(x),'-',label='$R_{0}$ of Settlement 2')
plt.text(0.7,15.1,'$R_{0}$ of settlement2',fontsize=15)
plt.text(0.7,8.1,'$R_{0}$ of settlement1',fontsize=15)
plt.text(0.7,11.65,'Average of $R_{0}$ of settlements',fontsize=15)
plt.plot(x,[8]*len(x),'-',label='$R_{0}$ of Settlement 1')
plt.errorbar(x,y,yerr=yerr,fmt='o',color='k')
#plt.legend()
plt.show()
