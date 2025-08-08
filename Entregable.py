#Ejercicio 1 - Análisis de datos financieros con Pandas y Numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("__________Datos de CSV__________")
res = pd.read_csv("finanza1.csv")
print(res)

print("__________Datos Limpiados__________")
res.dropna(inplace=True) #Eliminamos las filas vacias que esten vacios (NaN)
print(res)

print("__________Calcular estadisticas basicas__________")
print("Descriptiva:")
print(res.describe())

print("Desviación de la utilidad:", np.std(res['Utilidad']))
print("Utilidad total:", res['Utilidad'].sum())
print("Promedio de ingresos:", res['Ingresos'].mean())#sumamos la ingreso y lo divimos
print("Promedio de gastos:", res['Gastos'].mean())

print("__________Vizualizaciones inciales con Matplotlib y Seaborn__________")
plt.plot(res['Fecha'], res['Ingresos'], label='ingreso')
plt.plot(res['Fecha'], res['Gastos'], label='gastos')
plt.plot(res['Fecha'], res['Utilidad'], label='utilidad')
plt.title('evolucion de ingresos, gastos, utilidad, ')
plt.xlabel('Fecha')
plt.ylabel('Soles')
plt.show()

sns.histplot(res['Utilidad'], kde=True)
plt.title("cuántos meses caen en cada rango")
plt.xlabel("Utilidad")
plt.ylabel("Frecuencia")
plt.show()

#Ejercicio 2 – Predicción de tendencias de mercado con Scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

res = pd.read_csv("finanza1.csv")

res['Tendencia'] = (res['Utilidad'].shift(-1) > res['Utilidad']).astype(int)
res = res.dropna()

X = res[['Ingresos', 'Gastos', 'Utilidad']]
y = res['Tendencia']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
modelo = RandomForestClassifier().fit(X_train, y_train)

print("Precisión", modelo.score(X_test, y_test))

#Ejercicio 3 - Predicción con redes neuronales usando Keras/TensorFlow

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = pd.read_csv("finanza.csv")
data = df['Utilidad'].values.reshape(-1, 1)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

X, y = [], []
pasos = 5
for i in range(len(data_scaled) - pasos):
    X.append(data_scaled[i:i+pasos, 0])
    y.append(data_scaled[i+pasos, 0])
X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential([
    LSTM(50, activation='relu', input_shape=(pasos, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=20, batch_size=8, verbose=0)

# Predecir
pred = model.predict(X_test)
pred_inv = scaler.inverse_transform(pred)

print("Predicciones ejercicio 3 (primeros 5 valores):", pred_inv[:5].flatten())

#Ejercicio 4 - Análisis de reportes financieros con NLTK (NLP)
