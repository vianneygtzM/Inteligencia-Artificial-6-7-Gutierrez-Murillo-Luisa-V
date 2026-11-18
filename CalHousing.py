from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

california = fetch_california_housing(as_frame=True)

#Definir las variables X (independiente) y Y (dependiente)
#se toma el ingreso medio ('MedInc') como característica y el valor de la casa ('MedHouseVal' / target)
#usar un subconjunto de 50 registros para el ejercicio práctico
X = california.data[['MedInc']].values[:50]
Y = california.target.values[:50]

#dividir los datos en conjunto de entrenamiento (80%) y de prueba (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Creamos e instanciamos el modelo de Regresión Lineal
modelo = LinearRegression()

#entrenar el modelo con los datos de entrenamiento
modelo.fit(X_train, Y_train)

#Realizar predicciones utilizando el conjunto de prueba
Y_pred = modelo.predict(X_test)

#Evaluar el rendimiento del modelo
print("--- Resultados del Modelo ---")
print(f"Coeficiente (Pendiente m): {modelo.coef_[0]:.4f}")
print(f"Intercepto (b): {modelo.intercept_:.4f}")
print(f"R cuadrada (R²): {r2_score(Y_test, Y_pred):.4f}")
print(f"Error Cuadrático Medio (MSE): {mean_squared_error(Y_test, Y_pred):.4f}")

# Graficar los resultados
plt.scatter(X, Y, color='blue', label='Datos reales (50 muestras)')
plt.plot(X_test, Y_pred, color='red', linewidth=2, label='Línea de Regresión')
plt.xlabel('Ingreso Medio (MedInc)')
plt.ylabel('Valor Medio de la Casa (Target)')
plt.title('Regresión Lineal Simple - California Housing')
plt.legend()
plt.show()