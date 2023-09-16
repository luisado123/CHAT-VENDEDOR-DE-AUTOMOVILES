import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random

# Inicializar el lematizador
lemmatizer = WordNetLemmatizer()

# Cargar el modelo previamente entrenado
model = load_model('chatbot_model.h5')

# Cargar los datos de intenciones y palabras
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def clean_up_sentence(sentence):
    # Tokenizar el patrón - dividir las palabras en un array
    sentence_words = nltk.word_tokenize(sentence)
    
    # Lematizar cada palabra y crear forma corta para cada palabra
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    
    return sentence_words

def bow(sentence, words, show_details=True):
    # Tokenizar el patrón
    sentence_words = clean_up_sentence(sentence)
    
    # Crear bolsa de palabras: matriz de palabras, matriz de vocabulario
    bag = [0] * len(words)
    
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # Asignar 1 si la palabra actual está en la posición del vocabulario
                bag[i] = 1
                if show_details:
                    print("encontrado en la bolsa: %s" % w)
                    
    return np.array(bag)

def predict_class(sentence, model):
    # Filtrar predicciones por encima de un umbral
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # Ordenar por fuerza de probabilidad
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append({
            "intent": classes[r[0]],
            "probability": str(r[1])
        })
        
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    #list_of_intents = intents_json['intents']
    list_of_intents = json.loads(open('intents.json', 'r', encoding='utf-8').read())['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    if not ints:  # Si no se encuentra ninguna intención
        return "Es probable que quieras conversar sobre algo para lo que no estoy capacitado. Por favor, revisa la lista de temas en los que puedo ayudar."
    
    res = getResponse(ints, intents)
    return res


import tkinter
from tkinter import *

def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)
    
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))
        
        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

base = Tk()
base.title("Chatbot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

# Crear ventana de chat
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial")
ChatLog.config(state=DISABLED)

# Enlazar scrollbar a la ventana de chat
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# Crear botón para enviar mensajes
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff', command=send)

# Crear el cuadro para ingresar mensajes
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")

scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()