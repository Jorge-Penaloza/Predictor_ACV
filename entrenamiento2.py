import pandas as pd
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

df = pd.read_csv('G:/Mi unidad/Universidad/Proyecto de titulo/PT/dataset1.1.csv', sep=";") 
# El resto de tu preprocesamiento de datos aquí ...

# Normalización de datos
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

# Guardar el objeto min_max_scaler
joblib.dump(min_max_scaler, 'min_max_scaler.pkl')

# Separación de datos de entrenamiento y validación
X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y, test_size=0.3)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Aplicar regresión logística en lugar de la red neuronal
logreg = LogisticRegression()
logreg.fit(X_train_scaled, Y_train)

Y_pred = logreg.predict(X_test_scaled)
accuracy = accuracy_score(Y_test, Y_pred)

print("Accuracy of logistic regression: ", accuracy)

#Guardar el modelo
joblib.dump(logreg, 'logistic_model.pkl')  # Crea un archivo 'logistic_model.pkl'
