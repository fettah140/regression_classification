from time import *
t_départ=time()
#1.	Chargement des librairies Python
import pandas as pd
import numpy as np

#2.	Chargement des données
dataset = pd.read_csv('book.csv')
dataset.head()
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values


#3.les données en ensembles d'entraînement et de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=0)

#4.	Mise à l'échelle des fonctionnalités
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#5.	Entraînement de l'algorithme
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#6.	Évaluation de l'algorithme
print('prédction de fiabilité:',y_pred[0:1] )
print("Types d'algorithme:Classification & Régression ")
print("nombre de paramètre: 2.0")
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
t_fin=time()
print("tempsd'exécution:",t_fin-t_départ,"secondes")





