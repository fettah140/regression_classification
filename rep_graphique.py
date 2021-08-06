

#7.Représentation graphique
import matplotlib.pyplot as plt
n_neighbors=[1,2,3]
Mean_Squared_Error=[0,1.071,10.432]
plt.scatter(n_neighbors,Mean_Squared_Error)
plt.xlabel("n_neighbors")
plt.ylabel("Mean_Squared_Error")
plt.plot(n_neighbors,Mean_Squared_Error,color="green",ls="--")
plt.grid()
plt.show()

#7.Représentation graphique « pour le Random Forest »
import matplotlib.pyplot as plt
n_estimateurs=[10,50 ,100 ,150 ,200, 250,300,400,500 ]
Mean_Squared_Error=[1.912,2.232,1.690,1.561,1.592,1.368,1.163,1.103,0.949]
plt.scatter(n_estimateurs,Mean_Squared_Error)
plt.xlabel("n_estimateurs")
plt.ylabel("Mean_Squared_Error")
plt.plot(n_estimateurs,Mean_Squared_Error,color="blue",ls="--")
plt.grid()
plt.show()
