import speech_recognition as sr
import pyaudio
r= sr.Recognizer()
#print(sr.Microphone.list_microphone_names()[7])
mic = sr.Microphone(device_index=7)
with mic as source:
    print('Dime algo...')
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)
    voice = r.recognize_google(audio, language='es-MX')
    print(voice)