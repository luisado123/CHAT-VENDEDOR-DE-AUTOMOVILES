import nltk
nltk.download('wordnet')  # Descargar el recurso WordNet

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import tensorflow as tf
import random
words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json', 'r', encoding='utf-8').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenizar cada palabra
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Agregar documentos al corpus
        documents.append((w, intent['tag']))
        # Agregar a nuestra lista de clases
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# Lematizar, convertir a minúsculas y eliminar palabras duplicadas
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Ordenar las clases
classes = sorted(list(set(classes)))

print("Palabras:", words)
print("Clases:", classes)
# Combinación de patrones e intenciones para crear documentos
#documents = [(pattern, intent) for pattern, intent in zip(words, classes)]
for word, intent in zip(words, classes):
    documents.append((word, intent))
#for pattern, tag in documents:
#    print("Patrón preprocesado:", pattern)
#    print("Etiqueta:", tag)
#    print("----")

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Guardar las listas en archivos pickle
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)

with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

# Crear nuestros datos de entrenamiento
training = []
# Crear un arreglo vacío para nuestras salidas
output_empty = [0] * len(classes)
# Generar conjunto de entrenamiento: bolsa de palabras para cada sentencia
for doc in documents:
    # Inicializar nuestra bolsa de palabras
    bag = []

    # Lista de palabras tokenizadas para el patrón
    pattern_words = doc[0]

    # Lematizar cada palabra y crear palabra base para representar palabras relacionadas
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # Crear nuestra bolsa de palabras con '1' si se encuentra la palabra en el patrón actual
    for w in words:
        if w in pattern_words:
            bag.append(1)
        else:
            bag.append(0)

    # Crear la lista de salida (etiqueta)
    output_row = [0] * len(classes)
    output_row[classes.index(doc[1])] = 1

    # Agregar la bolsa de palabras y la etiqueta a los datos de entrenamiento
    training.append([bag, output_row])

# Mezclar nuestras características
random.shuffle(training)

# Separar las bolsas de palabras y las etiquetas
train_x = np.array([bag for bag, _ in training])
train_y = np.array([output_row for _, output_row in training])

print("Datos de entrenamiento creados")




# Crear el modelo: 3 capas. Primera capa con 128 neuronas, segunda capa con 64 neuronas
# y la tercera capa de salida contiene un número de neuronas igual al número de intenciones
# para predecir la intención de salida utilizando softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compilar el modelo. El descenso de gradiente estocástico con aceleración de Nesterov
# proporciona buenos resultados para este modelo
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenar y guardar el modelo
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
#for epoch in range(1, 200):
#    print(f"Época {epoch}/{len(hist.history['loss'])}")
#    print(f" - Pérdida de entrenamiento: {hist.history['loss'][epoch-1]}")
#    print(f" - Precisión de entrenamiento: {hist.history['accuracy'][epoch-1]}")
model.save('chatbot_model.h5', hist)
print("Modelo creado")
