import matplotlib.pyplot as plt
import pandas

df=pandas.read_csv("values.csv")
plt.clf()
plt.figure(figsize=(30,15))
plt.plot(df["transfer_values"],df["total_ro"],'bo',label='actual total0')
plt.plot(df["transfer_values"],df["city1_ro"],'k*',label='city1 cal')
plt.plot(df["transfer_values"],df["city2_ro"],'mv',label='city2 cal')
plt.plot(df["transfer_values"],df["mean"],'g-',label='mean of ro')
plt.plot(df["transfer_values"],df["max"],'r-',label='max of the ro')
plt.plot(df["transfer_values"],df["min"],'k-',label='min of the ro')
plt.legend(loc='best')
plt.title("$\sigma=0.8,\gamma=0.1$")
plt.savefig('plot_city2.png', format='png', orientation='landscape')
plt.close()
