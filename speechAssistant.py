import speech_recognition as sr
import pyaudio
import sys
from playsound import playsound 
import os
import random
from gtts import gTTS 

r= sr.Recognizer()
mic = sr.Microphone()
def hablar(au_string):
    tts = gTTS(text=au_string, lang='es') #slow= True
    r = random.randint(1, 1000000)
    audio_file = 'audio-' + str(r) + '.mp3'
    tts.save(audio_file)
    playsound(audio_file)
    print(au_string)
    os.remove(audio_file)

def grabarAudio():
    
    #Asi podemos reconocer los microfonos disponibles, es este caso 7 es el de default
    #print(sr.Microphone.list_microphone_names()[7])
    voice = ''
    #print('falle...')
    with mic as source:
        
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        #print('falle...')
        try:
            voice = r.recognize_google(audio, language='es-MX')
        except sr.UnknownValueError:
            print("Lo siento, no te entendi")
        except sr.RequestError:
            print("Lo siento, mi servicio de habla fallo")
        return voice

# if 'palabra clave' in 'frase entrada'
#print('Te escucho...')
#voice = grabarAudio()
#print(voice)
#hablar(voice)
def existe(palabrasValidas):
    for palabra in palabrasValidas:
        if palabra in entrada:
            return True

def responder(entrada):
    #1 Dar la bienvenida
    if existe(['hola', 'buenos días', 'buenas tardes', 'buenas noches']):
        hablar(fraseBienvenido)
    
    #2Buscar todos los objetos que tengo al frente
    if "que tengo al frente" in entrada:
        
        fraseDeteccion = fraseObjetosAlFrente
        for objeto in objetos:
            fraseDeteccion = fraseDeteccion + ' ' + objeto + ','
        hablar(fraseDeteccion)

    #3 Buscar objetos y sus distancias 
    if existe(['distancia', 'lejos']):
        fraseDeteccion = "Tienes"
        for objeto in objetosDistancia.keys():
            fraseDeteccion = fraseDeteccion + "{} {} a {} metros, ".format(objetosDistancia[objeto][0],objeto,objetosDistancia[objeto][1] )
        hablar(fraseDeteccion)
    #4 Buscar objetos a la derecha 

    #5 Buscar objetos a la izquierda 



##Implementacion asistente de voz
objetos = ['monitor', 'persona', 'gato', 'teclado'] #Util para funcion #2
objetosDistancia = {"monitor": [2, 125.2],
                    "persona": [1, 325.2],
                    "gato": [1, 25.2]}
nombreUsuario = 'javier'
nombreAsistente = 'victoria'
fraseBienvenido ="Hola {}, soy {}, ¿en qué te puedo ayudar?".format(nombreUsuario, nombreAsistente)
fraseDespedida = "Adiós " + nombreUsuario
fraseObjetosAlFrente = 'Al frente tienes'
try:
    while(1): 
        print('Estoy escuchando')
        
        entrada = grabarAudio()
        entrada = entrada.lower()
        if nombreAsistente in entrada: 
            responder(entrada)
            print(entrada)
        else:
            print("Estoy esperando")
        #if ('adiós' in entrada):
         #   hablar(fraseDespedida)
          #  sys.exit()
        #print(entrada)
        
       # if 'hola victoria' in entrada:
        #    hablar(fraseBienvenido)
        #else: 
         #   print('Sigo esperando...')
except sr.UnknownValueError: 
    print('No escucho nada...')
except sr.RequestError as e:
    print("Ocurrio un error ..." + e)
