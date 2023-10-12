from keras.models import load_model 
import joblib
import numpy as np

# Cargar el modelo y MinMaxScaler
model = load_model('my_model.h5')
min_max_scaler = joblib.load('min_max_scaler.pkl')

# Supongamos que recibimos 11 entradas de algún tipo de interfaz de usuario
inputs = [0, 67, 0, 1, 1, 3, 0, 228.69, 36.6, 3]  # Reemplazar con los valores reales

# Asume que las entradas son unidimensionales
inputs = np.array(inputs).reshape(1, -1)

# Normalizar las entradas
inputs = min_max_scaler.transform(inputs)

# Hacer una predicción con el modelo
predictions = model.predict(inputs)

# Imprimir las predicciones
print(predictions)
#9046;Male;67;0;1;Yes;Private;Urban;228,69;36.6;formerly smoked;1
#Dado a la baja de los resultamos se planteo eliminar carcteristicas que no influyen en la prediccion