import speech_recognition as sr
import pyaudio

from playsound import playsound 
import os
import random
from gtts import gTTS 


def hablar(au_string):
    tts = gTTS(text=au_string, lang='es') #slow= True
    r = random.randint(1, 1000000)
    audio_file = 'audio-' + str(r) + '.mp3'
    tts.save(audio_file)
    playsound(audio_file)
    print(au_string)
    os.remove(audio_file)

def grabarAudio():
    r= sr.Recognizer()
    #Asi podemos reconocer los microfonos disponibles, es este caso 7 es el de default
    #print(sr.Microphone.list_microphone_names()[7])
    mic = sr.Microphone(device_index=7)
    voice = ''
    with mic as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        try:
            voice = r.recognize_google(audio, language='es-MX')
        except sr.UnknownValueError:
            print("Lo siento, no te entendi")
        except sr.RequestError:
            print("Lo siento, mi servicio de habla fallo")
        return voice

# if 'palabra clave' in 'frase entrada'
print('Te escucho...')
voice = grabarAudio()
print(voice)
hablar(voice)