import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Crear y cargar un conjunto de datos de texto
# Para simplificar, crearemos un pequeño DataFrame de Pandas.
# En un proyecto real, cargarías un archivo CSV o JSON.
data = {
    'review': [
        "me encanta esta pelicula, es genial",
        "la pelicula es aburrida y terrible",
        "una de las mejores peliculas que he visto",
        "no me gusto la trama, fue muy predecible",
        "que actuacion tan espectacular, me emociono mucho",
        "fue una experiencia horrible, no la recomiendo"
    ],
    'sentiment': [
        'positivo',
        'negativo',
        'positivo',
        'negativo',
        'positivo',
        'negativo'
    ]
}
df = pd.DataFrame(data)

print("Datos de ejemplo:")
print(df)

# 2. Separar las características (X) y las etiquetas (y)
X = df['review']  # Las reseñas de texto
y = df['sentiment']  # El sentimiento (positivo/negativo)

# 3. Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# 4. Vectorizar los datos de texto
# El modelo no puede procesar texto directamente,
# así que lo convertimos en una matriz de "frecuencia de palabras".
vectorizer = CountVectorizer()

# Aprende el vocabulario y transforma los datos de entrenamiento
X_train_vectorized = vectorizer.fit_transform(X_train)

# Transforma los datos de prueba usando el mismo vocabulario aprendido
X_test_vectorized = vectorizer.transform(X_test)

print("\nForma de los datos vectorizados (entrenamiento):", X_train_vectorized.shape)
print("Ejemplo de vectorización (primera reseña):")
print(X_train_vectorized.toarray()[0])

# 5. Crear y entrenar un modelo
# MultinomialNB es un algoritmo ideal para datos de frecuencia (conteo de palabras).
model = MultinomialNB()
print("\nEntrenando el modelo...")
model.fit(X_train_vectorized, y_train)

# 6. Hacer predicciones
y_pred = model.predict(X_test_vectorized)
print("\nPredicciones del modelo:", y_pred)
print("Valores reales:", y_test.values)

# 7. Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo: {accuracy:.2f}")
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# 8. Ejemplo de predicción con un nuevo texto
new_review = ["espectacular pelicula, me gusto mucho"]
new_review_vectorized = vectorizer.transform(new_review)

prediction = model.predict(new_review_vectorized)
print(f"\nLa reseña '{new_review[0]}' fue clasificada como: {prediction[0]}")