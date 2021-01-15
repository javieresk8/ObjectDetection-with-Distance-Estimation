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

##Datos iniciales para medir distancia de un celular a la camara (CM)
DISTANCIA_INIC_CELULAR = 50 
PIXELES_DIST_INIC_CELULAR = 380 #Tomado del video
ANCHO_REAL_CELULAR = 13.5


def Convertir_RGB(img):
    # Convertir Blue, green, red a Red, green, blue
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def Convertir_BGR(img):
    # Convertir red, blue, green a Blue, green, red
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img

def dibujarLinea(imagen, puntoInicial, puntoFinal, color, grosor):
    cv2.line(imagen, puntoInicial, puntoFinal, color, grosor)

def obtenerCentroide(x1, y1, x2, y2):
    return ((x1 + x2)/2, (y1 + y2) / 2 )

def obtenerAnchoCaja(x1, x2):
    return round(float(x2) - float(x1), 2)

def calcularFocal(anchoPixeles, distInicial, anchoReal):
    return (anchoPixeles*distInicial)/anchoReal


#def calcularDistancia(anchoPixeles, distInicial, anchoReal, anchoVariable):
   #la distancia se calcula para un objeto como un celular
#    focal = calcularFocal(anchoPixeles, distInicial, anchoReal)
#    return anchoReal * focal / anchoVariable

def calcularDistancia(anchoVariable, distInic, pixInic):
    return round(pixInic*distInic/anchoVariable, 2)

if __name__ == "__main__":
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
        #Aqui toma el video de la webcam
        cap = cv2.VideoCapture(0)
        out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,960))
    else:
        #Toma el video que este en el directorio
        cap = cv2.VideoCapture(opt.directorio_video)
        out = cv2.VideoWriter('outp.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,960))
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    a=[]

    #Comienza a procesar las imagenes del video
    while cap:
        #Abrimos la primera imagen del video
        ret, frame = cap.read()
        if ret is False:
            break
        #Creamos e; frame 
        frame = cv2.resize(frame, (1280, 960), interpolation=cv2.INTER_CUBIC)
        #LA imagen viene en Blue, Green, Red y la convertimos a RGB que es la entrada que requiere el modelo
        #Preparamos la imagen de aucerdo a los tensores para que el modelo pueda entenderlo
        RGBimg=Convertir_RGB(frame)
        imgTensor = transforms.ToTensor()(RGBimg)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, 416)
        imgTensor = imgTensor.unsqueeze(0)
        imgTensor = Variable(imgTensor.type(Tensor))


        with torch.no_grad():
            #Le pasamos el tensor con la imagen preparada al modelo
            detections = model(imgTensor)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        #las detecciones nos van a dar el numero de la etiqueta (id)
        #Dibujamos la caja, obtenemos coordenadas, certeza y nombre de la clase 
        for detection in detections:
            if detection is not None:
                detection = rescale_boxes(detection, opt.img_size, RGBimg.shape[:2])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    #Calculamos dimensiones de la caja
                    ancho_caja = x2 - x1
                    altura_caja = y2 - y1
                    color = [int(c) for c in colors[int(cls_pred)]]
                    print("Se detectó {} en X1: {}, Y1: {}, X2: {}, Y2: {}".format(classes[int(cls_pred)], x1, y1, x2, y2))
                    frame = cv2.rectangle(frame, (x1, y1 + altura_caja), (x2, y1), color, 5)
                    cv2.putText(frame, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5)# Nombre de la clase detectada
                    cv2.putText(frame, str("%.2f" % float(conf)), (x2, y2 - altura_caja), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 5) # Certeza de prediccion de la clase
                    #Dibujamos la linea en el centro de cada imagen
                    centroide = obtenerCentroide(x1, y1, x2, y2)
                    dibujarLinea(frame, (640,960), centroide, (0,255,0), 3)
                    
                    #CALCULAMOS LA DISTANCIA
                    ancho_caja_trans = obtenerAnchoCaja(x1, x2)
                    distancia = calcularDistancia(ancho_caja_trans, DISTANCIA_INIC_CELULAR, PIXELES_DIST_INIC_CELULAR)
                    #Escribimos la distancia
                    cv2.putText(frame, str(distancia) + "cm", centroide, 0, 1, (255,0,0), 3, cv2.LINE_AA)
                
        #
        #Convertimos de vuelta a BGR para que cv2 pueda desplegarlo en los colores correctos
        
        if opt.webcam==1:
            cv2.imshow('frame', Convertir_BGR(RGBimg))
            out.write(RGBimg)
        else:
            out.write(Convertir_BGR(RGBimg))
            cv2.imshow('frame', RGBimg)
        #cv2.waitKey(0)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    out.release()
    cap.release()
    cv2.destroyAllWindows()
