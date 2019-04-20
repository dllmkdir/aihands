"""
Utilization:
the variable "letra" is the image container, modify it for you letter case
Spacebar -> shoot a photo
q -> Leave the program
"""
import numpy as np
import cv2
import random
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
letra = 'D'
dir_general = './dataset'
dir_letra = dir_general + '/' + letra
# Creando carpetas para las imagenes
if not os.path.isdir(dir_general) :
	os.mkdir(dir_general)
	print("Directorio " , dir_general ,  " creado ")
else:
	print("Directorio " , dir_general ,  " disponible para captura")

if not os.path.isdir(dir_letra) :
	os.mkdir(dir_letra)
	print("Directorio " , dir_letra ,  " creado ")
else:
	print("Directorio " , dir_letra ,  " disponible para captura")



# initialize the camera
cam = cv2.VideoCapture(0) # 0 -> index of camera
nombre_img = letra+'0'
cnt = 0
pcnt = 0
#box coordinates
x,y,w,h = 100,100,300,300
print("---Ready For capture---")
while(True):
    ret, image = cam.read()
    box = cv2.rectangle(image ,(x-3, y-3), (x+w+3, y+h+3),(255,255,0),3)
    #image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)
    image_flipped = cv2.flip(image, 1)#flip image
    cv2.imshow('captura',image_flipped)

    if ret: # if frame is captured without any errors
        try:  # used try so that if user pressed other than the given key error will not be shown
            if (cv2.waitKey(1) & 0xff) == 32:#wait for space

                while(os.path.isfile(dir_letra+'/'+nombre_img+'.jpg')):
                    pcnt = 0 + cnt
                    cnt = cnt + 1
                    nombre_img = nombre_img.replace(str(pcnt),str(cnt))

                salpicado = np.zeros(image.shape,np.uint8) # Salpicando de sal y pimienta
                p = 0.001	# probablity of noise
                for i in range (image.shape[0]):
                    for j in range(image.shape[1]):
                        r = random.random()
                        if r < p/2:
                            salpicado[i][j] = 0,0,0
                        elif r < p:
                            salpicado[i][j] = 255,255,255
                        else:
                            salpicado[i][j] = image[i][j]
                #salpicado = cv2.resize(salpicado, (224, 224))
                #escalado = cv2.resize(salpicado, (224,224), interpolation = cv2.INTER_AREA)
                crop_img = cv2.flip(salpicado[y:y+h, x:x+w],1)
                cv2.imwrite((dir_letra+'/'+nombre_img+'.jpg'),crop_img) #save image
                print("image {}.jpg succesfully captured".format(nombre_img))
            else:
                pass
        except:
            break  # if user pressed a key other than the given key the loop will break
	# Wait for Escape Key  
    if (cv2.waitKey(10) & 0xff) == 113 :#wait for q
        break
cam.release()
cv2.destroyAllWindows()
print("---Capture finished---")