import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Cambia el nombre del archivo si tu dataset tiene otro nombre o extensión
df = pd.read_csv(r'C:\Users\isai5\OneDrive\Documentos\Github\databases\StressLevelDataset.csv')

# Visualizar la distribución de stress_level
plt.figure(figsize=(6,4))
sns.countplot(x='stress_level', data=df)
plt.title('Distribución de Stress Level')
plt.xlabel('Nivel de Estrés')
plt.ylabel('Cantidad')
plt.show()

# Seleccionar características numéricas (excepto stress_level)
X = df.drop('stress_level', axis=1).select_dtypes(include=['number'])
y = df['stress_level']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear modelo secuencial
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 clases: 0, 1, 2
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar modelo
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)

# Evaluar modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Precisión en test: {accuracy:.2f}')

# Graficar la curva de precisión
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.show()

# Matriz de confusión
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# Reporte de clasificación
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred_classes))

# Conclusión simple
print("Conclusión:")
if accuracy > 0.7:
    print("El modelo tiene un buen desempeño para predecir el nivel de estrés.")
else:
    print("El modelo puede mejorarse, prueba ajustando hiperparámetros o usando más datos.")