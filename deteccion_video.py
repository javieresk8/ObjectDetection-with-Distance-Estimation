from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
import os
import sys
import argparse
import cv2
from PIL import Image
import torch
from torch.autograd import Variable
import requests
from time import time


#Medidas tomadas de una altura de 131cm.
##Datos iniciales para medir distancia de una persona a la camara (CM)
DISTANCIA_INIC_PERSONA = 335 
AREA_INIC_PERSONA = 114 #Tomado del video


##Datos iniciales para medir distancia de un gato a la camara (CM)
DISTANCIA_INIC_GATO = 200 
AREA_INIC_GATO = 17 #Tomado del video

##Datos iniciales para medir distancia de una botella a la camara (CM)
DISTANCIA_INIC_BOTELLA = 37 
AREA_INIC_BOTELLA = 123 #Tomado del video

##Datos iniciales para medir distancia de una TAZA a la camara (CM)
DISTANCIA_INIC_TAZA = 37 
AREA_INIC_TAZA = 94 #Tomado del video

##Datos iniciales para medir distancia de una TENEDOR a la camara (CM)
DISTANCIA_INIC_TENEDOR = 37 
AREA_INIC_TENEDOR = 35 #Tomado del video

##Datos iniciales para medir distancia de una CUCHILLO a la camara (CM)
DISTANCIA_INIC_CUCHILLO = 37 
AREA_INIC_CUCHILLO = 60 #Tomado del video

##Datos iniciales para medir distancia de una CUCHARA a la camara (CM)
DISTANCIA_INIC_CUCHARA = 37 
AREA_INIC_CUCHARA = 85 #Tomado del video

##Datos iniciales para medir distancia de una SILLA a la camara (CM)
DISTANCIA_INIC_SILLA = 150 
AREA_INIC_SILLA = 240 #Tomado del video

##Datos iniciales para medir distancia de una CAMA a la camara (CM)
DISTANCIA_INIC_CAMA = 150 
AREA_INIC_CAMA = 830 #Tomado del video

##Datos iniciales para medir distancia de una INODORO a la camara (CM)
DISTANCIA_INIC_INODORO = 105 
AREA_INIC_INODORO = 320 #Tomado del video

##Datos iniciales para medir distancia de una MONITOR a la camara (CM)
DISTANCIA_INIC_MONITOR = 91 
AREA_INIC_MONITOR = 295 #Tomado del video

##Datos iniciales para medir distancia de una LAPTOP a la camara (CM)
DISTANCIA_INIC_LAPTOP = 75 
AREA_INIC_LAPTOP = 380 #Tomado del video

##Datos iniciales para medir distancia de una MOUSE a la camara (CM)
DISTANCIA_INIC_MOUSE = 37 
AREA_INIC_MOUSE = 50 #Tomado del video


##Datos iniciales para medir distancia de una TECLADO a la camara (CM)
DISTANCIA_INIC_TECLADO = 40 
AREA_INIC_TECLADO = 265 #Tomado del video

##Datos iniciales para medir distancia de un CELULAR a la camara (CM)
DISTANCIA_INIC_CELULAR = 37 
AREA_INIC_CELULAR = 106 #Tomado del video


##Datos iniciales para medir distancia de una LIBRO a la camara (CM)
DISTANCIA_INIC_LIBRO = 37 
AREA_INIC_LIBRO = 410 #Tomado del video

##Datos iniciales para medir distancia de una RELOJ a la camara (CM)
#DISTANCIA_INIC_RELOJ = 21 
#AREA_INIC_RELOJ = 53 #Tomado del video

#Creamos el diccionario que contiene todas las clases y las distancias 
objetosDeteccion = {'persona': [DISTANCIA_INIC_PERSONA, AREA_INIC_PERSONA], 
    'gato': [DISTANCIA_INIC_GATO, AREA_INIC_GATO], 
    'botella': [DISTANCIA_INIC_BOTELLA, AREA_INIC_BOTELLA],
    'taza': [DISTANCIA_INIC_TAZA, AREA_INIC_TAZA],
    'tenedor': [DISTANCIA_INIC_TENEDOR, AREA_INIC_TENEDOR],
    'cuchillo': [DISTANCIA_INIC_CUCHILLO, AREA_INIC_CUCHILLO],
    'cuchara': [DISTANCIA_INIC_CUCHARA, AREA_INIC_CUCHARA],
    'silla': [DISTANCIA_INIC_SILLA, AREA_INIC_SILLA],
    'cama': [DISTANCIA_INIC_CAMA, AREA_INIC_CAMA],
    'inodoro': [DISTANCIA_INIC_INODORO, AREA_INIC_INODORO],
    'monitor': [DISTANCIA_INIC_MONITOR, AREA_INIC_MONITOR],
    'laptop': [DISTANCIA_INIC_LAPTOP, AREA_INIC_LAPTOP],
    'mouse': [DISTANCIA_INIC_MOUSE, AREA_INIC_MOUSE],
    'teclado': [DISTANCIA_INIC_TECLADO, AREA_INIC_TECLADO],
    'celular': [DISTANCIA_INIC_CELULAR, AREA_INIC_CELULAR],
    'libro': [DISTANCIA_INIC_LIBRO, AREA_INIC_LIBRO]}


def Convertir_RGB(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def Convertir_BGR(img):
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img

def sumarTuplas(tupla1, tupla2):
    return tuple(a + b for a, b in zip(tupla1, tupla2))

def dibujarLinea(imagen, puntoInicial, puntoFinal, color, grosor):
    cv2.line(imagen, puntoInicial, puntoFinal, color, grosor)

def obtenerCentroide(x1, y1, x2, y2):
    return ((x1 + x2)/2, (y1 + y2) / 2 )

def obtenerAnchoCaja(x1, x2):
    return round(float(x2) - float(x1), 2)

def obtenerAreaCaja(ancho, alto):
    return round((float(ancho)* float(alto)/1000), 2)

def calcularFocal(anchoPixeles, distInicial, anchoReal):
    return (anchoPixeles*distInicial)/anchoReal



def calcularDistancia(areaVariable, distInic, areaInic):
    return round(areaInic*distInic/areaVariable, 2) 


def detectarObjetosYOLO():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--webcam", type=int, default=1,  help="Is the video processed video? 1 = Yes, 0 == no" )
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--directorio_video", type=str, help="Directorio al video")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)
    #Si encuentra una GPU lo usa sino usa CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)


    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  
    classes = load_classes(opt.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    if opt.webcam==1:
        #URL de IP WEBCAM
        url = 'http://192.168.0.2:8080/shot.jpg'
        #Aqui toma el video de la webcam
        cap = cv2.VideoCapture(0) #No cambiamos en ningun caso
        out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,960))
    else:
        #Toma el video que este en el directorio
        cap = cv2.VideoCapture(opt.directorio_video)
        out = cv2.VideoWriter('outp.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,960))
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    a=[]
    # Tomamos el tiempo para que dure solo 3 segundos la captacion de imagenes
    tiempo_inicial = time()
    tiempo_final_temp = 0
    tiempo_ejecucion = 0
    #Comienza a procesar las imagenes del video
    #Declaramos el diccionario que va a tener todas las distancias y los objetos detectados 
    medidasObjetosDetectados = []
    
    while cap:
        ret, frame = cap.read()
        if ret is False:
            break
        #Tomamos la imagen de IP WEbcam
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        frame = cv2.imdecode(img_arr, -1)
        frame = cv2.resize(frame, (1280, 960), interpolation=cv2.INTER_CUBIC)
        RGBimg=Convertir_RGB(frame)
        imgTensor = transforms.ToTensor()(RGBimg)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, 416)
        imgTensor = imgTensor.unsqueeze(0)
        imgTensor = Variable(imgTensor.type(Tensor))


        with torch.no_grad():
            detections = model(imgTensor)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        #las detecciones nos van a dar el numero de la etiqueta (id)
        #Dibujamos la caja, obtenemos coordenadas, certeza y nombre de la clase 
        for detection in detections:
            if detection is not None:
                detection = rescale_boxes(detection, opt.img_size, RGBimg.shape[:2])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    if classes[int(cls_pred)] == '':
                        continue
                    else:
                        ancho_caja = x2 - x1
                        altura_caja = y2 - y1
                        color = [int(c) for c in colors[int(cls_pred)]]
                        print("Se detectó {} en X1= {}, Y1= {}, X2= {}, Y2= {}".format(classes[int(cls_pred)], x1, y1, x2, y2))
                        frame = cv2.rectangle(frame, (x1, y1 + altura_caja), (x2, y1), color, 5)
                        cv2.putText(frame, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5)# Nombre de la clase detectada
                        cv2.putText(frame, str("%.2f" % float(conf)), (x2, y2 - altura_caja), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 5) # Certeza de prediccion de la clase
                        centroide = obtenerCentroide(x1, y1, x2, y2)
                        dibujarLinea(frame, (640,960), centroide, (0,255,0), 3)
                        distancia = 0
                        area_caja_trans = obtenerAreaCaja(ancho_caja,altura_caja)
                        claseDetectada = classes[int(cls_pred)]
                        distancia = calcularDistancia(area_caja_trans, objetosDeteccion[claseDetectada][0],objetosDeteccion[claseDetectada][1] )
                        area_ref = obtenerAreaCaja(ancho_caja, altura_caja) 
                        cv2.putText(frame, str(distancia) + "cm", centroide, 0, 1, (255,0,0), 3, cv2.LINE_AA)
                        
                        medidasObjetosDetectados.append([claseDetectada, distancia, centroide])
        if opt.webcam==1:
            cv2.imshow('frame', Convertir_BGR(RGBimg))
            out.write(RGBimg)
        else:
            out.write(Convertir_BGR(RGBimg))
            cv2.imshow('frame', RGBimg)
        #Para cerrar el frame tienen que haber pasado 3 segundos'
        
        tiempo_final_temp = time()
        tiempo_ejecucion = tiempo_final_temp -tiempo_inicial
        if cv2.waitKey(25) & 0xFF == ord('x') or tiempo_ejecucion > 3:
            break
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    print("================================Fin deteccion========================")

    #Se procede a calcular la media de la distancia en cada uno de los objetos detectados
    respuestaDeteccion = {}
    objetosDistanciaMedia = {}
    for objeto in medidasObjetosDetectados:
        nombreObjeto = objeto[0]
        distanciaObjeto = objeto[1]
        centroideObjeto = objeto[2]
        print("Se obtuvo {} a distancia {}".format(objeto[0], objeto[1]))
        #Metemos cada clase con clave = nombreClase valor= [sumaDistancias, cantidadDistancias]
        if objeto[0] not in respuestaDeteccion:
            respuestaDeteccion[objeto[0]] = [distanciaObjeto, 1, centroideObjeto]
        else:
            respuestaDeteccion[objeto[0]][0] = respuestaDeteccion[objeto[0]][0] + distanciaObjeto
            respuestaDeteccion[objeto[0]][1] = respuestaDeteccion[objeto[0]][1] + 1
            respuestaDeteccion[objeto[0]][2] = sumarTuplas(respuestaDeteccion[objeto[0]][2], centroideObjeto)
            
    for clase in respuestaDeteccion.keys():
        print("El total de la clase {} es {}cm de {} medidas, posicion = {}".format(clase, respuestaDeteccion[clase][0],respuestaDeteccion[clase][1],respuestaDeteccion[clase][2] ))

        #Armamos la respuesta final, tiene la distancia media en metros y la ubicacion en el eje horizontal de la camara 
        numeroDetecciones = respuestaDeteccion[objeto[0]][1]
        centroideTemp = respuestaDeteccion[clase][2]
        centroideMedio = (round(float(centroideTemp[0])/ numeroDetecciones), round(float(centroideTemp[1])/ numeroDetecciones))
        objetosDistanciaMedia[clase] = [round((respuestaDeteccion[clase][0] / respuestaDeteccion[clase][1])/100, 2), centroideMedio ]

        print("La distancia media de la clase {} es {} m, el centroide medio es = {} ".format(clase, objetosDistanciaMedia[clase][0], objetosDistanciaMedia[clase][1] ))
    return objetosDistanciaMedia