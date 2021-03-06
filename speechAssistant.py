import speech_recognition as sr
import pyaudio
import sys
from playsound import playsound 
import os
import random
from gtts import gTTS 
import deteccion_video
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
    
    #2Buscar todos los objetos que tengo en pantalla (FUNCIONANDO)
    if existe(['alrededor', 'todos los objetos']):
        objetosDetectados = deteccion_video.detectarObjetosYOLO()
        if len(objetosDetectados) > 0: 
            fraseDeteccion = fraseObjetosAlrededor
            for objeto in objetosDetectados.keys():
                fraseDeteccion = fraseDeteccion + " un {}, a {} metros, ".format( objeto, objetosDetectados[objeto][0])
            hablar(fraseDeteccion)
            print("============================LA CONSULTA ALREDEDOR SE HIZO CON EXITO===============")
        else: 
            hablar(fraseNoEncontreObjetos)

    #3 Buscar objetos en el segmento central
    if existe(['frente', 'alfrente', 'al frente', 'delante', 'adelante']):
        objetosDetectados = deteccion_video.detectarObjetosYOLO()
        if len(objetosDetectados) > 0:
            fraseDeteccion = fraseObjetosCentro
            for objeto in objetosDetectados.keys():
                if limiteSegmentoCentro > objetosDetectados[objeto][1][0] > limiteSegmentoIzquierda:
                    fraseDeteccion = fraseDeteccion + " un {}, a {} metros, ".format( objeto, objetosDetectados[objeto][0])
                
            hablar(fraseDeteccion)
            print("============================LA CONSULTA CENTRO SE HIZO CON EXITO===============")
        else: 
            hablar(fraseNoEncontreObjetos)

    #4 Buscar objetos a la izquierda (FUNCIONANDO) 
    if existe(['mi izquierda','la izquierda']):
        objetosDetectados = deteccion_video.detectarObjetosYOLO()
        if len(objetosDetectados) > 0:
            fraseDeteccion = fraseObjetosIzquierda
            for objeto in objetosDetectados.keys():
                if objetosDetectados[objeto][1][0] < limiteSegmentoIzquierda:
                    fraseDeteccion = fraseDeteccion + " un {}, a {} metros, ".format( objeto, objetosDetectados[objeto][0])
                
            hablar(fraseDeteccion)
            print("============================LA CONSULTA IZQUIERDA SE HIZO CON EXITO===============")
        else: 
            hablar(fraseNoEncontreObjetos)

    #5 Buscar objetos a la derecha (FUNCIONANDO) 
    if existe(['mi derecha','la derecha']):
        objetosDetectados = deteccion_video.detectarObjetosYOLO()
        if len(objetosDetectados) > 0:
            fraseDeteccion = fraseObjetosDerecha
            for objeto in objetosDetectados.keys():
                if objetosDetectados[objeto][1][0] > limiteSegmentoCentro:
                    fraseDeteccion = fraseDeteccion + " un {}, a {} metros, ".format( objeto, objetosDetectados[objeto][0])
                
            hablar(fraseDeteccion)
            print("============================LA CONSULTA DERECHA SE HIZO CON EXITO===============")
        else: 
            hablar(fraseNoEncontreObjetos)
    #6 Terminar ejecucion del asistente (FUNCIONANDO)
    if existe(['adiós', 'nos vemos', 'hasta luego', 'para', 'alto', 'detente', 'chao']):
        hablar(fraseDespedida)
        quit()
    
    #7 Preguntar si un objeto esta al frente, responde SI/NO y la distancia (FUNCIONANDO)
    if existe(['busca','hay algún', 'hay alguna']):
        objetosDetectados = deteccion_video.detectarObjetosYOLO()
        palabras = entrada.split()
        bandera = True
        
        for palabra in palabras:
            if palabra in objetosDetectados.keys():
                fraseDeteccion = "Si, encontré un {} a {} metros".format(palabra, objetosDetectados[palabra][0]) 
                hablar(fraseDeteccion)    
                bandera = False
        if bandera:
            fraseDeteccion = "No encontré ese objeto"
            hablar(fraseDeteccion)
        print("============================LA CONSULTA DERECHA SE HIZO CON EXITO===============")
    #8 Pregunta cuantos objetos de una clase hay

    #9 Pregunta el orden de los objetos que tiene alrededor 
     
    

##Implementacion asistente de voz
objetos = ['monitor', 'persona', 'gato', 'teclado'] #Util para funcion #2
objetosDistancia = {"monitor": [2, 125.2],
                    "persona": [1, 325.2],
                    "gato": [1, 25.2]}
nombreUsuario = 'javier'
nombreAsistente = 'victoria'
fraseBienvenido ="Hola {}, soy {}, ¿en qué te puedo ayudar?".format(nombreUsuario, nombreAsistente)
fraseDespedida = "Adiós " + nombreUsuario
fraseObjetosAlrededor = 'A tu alrededor tienes'
fraseObjetosDerecha = 'A tu derecha tienes'
fraseObjetosIzquierda = 'A tu izquierda tienes'
fraseObjetosCentro = 'Al frente tienes '
fraseNoEncontreObjetos = 'Lo siento, no encontré objetos que conozco'
fraseDespedida = "Adios " + nombreUsuario
limiteSegmentoIzquierda = 240
limiteSegmentoCentro = 900

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
