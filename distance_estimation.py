import numpy as np 
import cv2
import imutils
from imutils import paths #para cargar imagenes disponibles del directorio

#para encontrar el objeto que se va a medir
def buscarMarca(image):
    #convertir la imagen a escala de grises, desenfocarlo y detectar bordes
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #quitamos el ruido
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key = cv2.contourArea)
	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)

#aqui tomamos el ancho
def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth