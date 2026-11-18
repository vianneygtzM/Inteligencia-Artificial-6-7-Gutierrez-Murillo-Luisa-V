import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

#Datos de entrenamiento (Celsius y sus equivalentes exactos en Fahrenheit)
X_celsius = np.array([-40, -20, 0, 10, 20, 30, 40, 100]).reshape(-1, 1)
y_fahrenheit = X_celsius.ravel() * 1.8 + 32

#Escalado de datos (fundamental para que el descenso de gradiente converja rápido y bien)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_celsius)

#Entrenar la neurona lineal (Regresión con Descenso de Gradiente Estocástico)
regressor = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='invscaling', eta0=0.01, random_state=42)
regressor.fit(X_scaled, y_fahrenheit)

#Probar predicciones de valores continuos
test_celsius = np.array([15, 25, 37]).reshape(-1, 1)
test_scaled = scaler.transform(test_celsius)
predictions = regressor.predict(test_scaled)

print("\n--- RESULTADOS REGRESIÓN (NEURONA LINEAL) ---")
for c, f_pred in zip(test_celsius.ravel(), predictions):
    f_real = c * 1.8 + 32
    print(f"{c}°C -> Predicho por la neurona: {f_pred:.2f}°F | Real: {f_real:.2f}°F")