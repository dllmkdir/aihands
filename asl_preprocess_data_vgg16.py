# agrega liberías: keras, numpy, random, etc
from keras.utils import np_utils
from random import shuffle
import numpy as np
import random
import cv2
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# encuentra el directorio raíz
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Diccionario de clases para el reconocimiento de 3 letras: A, B y C
class_dic = {"A": 0, "B": 1, "C": 2}


image_list = []
image_class = []

# directorio temporal
path = "asl_img"

# directortio donde se encuentran las imágenes
data_folder_path = "asl_img"

# lista directorios de letras en orden alfabético
alphabet_dir = os.listdir(data_folder_path)

# contadore de número de clases o imágenes encontradas
class_count = {'A': 0, 'B': 0, 'C': 0}
# variable de almacenamiento de datos de entrenamiento
X = []
Y = []
# variables de almacenamiento de datos clasificados de validación
X_val = []
Y_val = []
# variables de almacenamiento de datos clasificados de prueba
X_test = []
Y_test = []
unique_list = []
for alpha_dir in alphabet_dir:
    temp_alphabet_dir = os.path.join(ROOT_DIR, data_folder_path, alpha_dir)
    print("Lectura en directorio {}".format(temp_alphabet_dir))
    for root, dirs, files in os.walk(temp_alphabet_dir):
        for file_name in files:
            #  print(file_name)
            # filename = "a1.jpg" label ="a"
            # filename = "b1.jpg" label ="b"
            label = file_name[0]
            if label not in unique_list:
                print(label)
                unique_list.append(label)
            path = os.path.join(temp_alphabet_dir, file_name)
            image = cv2.imread(path)  # Lectura de la imagen
            resized_image = cv2.resize(image, (224, 224))  # se provee de un resize
            # entrenamiento del modelo: 6000 objetos
            if class_count[label] < 2000:

                X.append(resized_image)
                Y.append(class_dic[label])
                class_count[label] += 1
            # validacion del modelo: 2250
            elif class_count[label] >= 2000 and class_count[label] < 2750:

                X_val.append(resized_image)
                Y_val.append(class_dic[label])
                class_count[label] += 1
            # pruebas del modelo: 750 objetos
            else:
                X_test.append(resized_image)
                Y_test.append(class_dic[label])

print(len(unique_list))

Y = np_utils.to_categorical(Y)
Y_val = np_utils.to_categorical(Y_val)
Y_test = np_utils.to_categorical(Y_test)

numpy_dir = "Numpy"
# se salva el entrenamiento
np.save(numpy_dir+'/train_set.npy', X)
np.save(numpy_dir+'/train_classes.npy', Y)
# se salva la validación
np.save(numpy_dir+'/validation_set.npy', X_val)
np.save(numpy_dir+'/validation_classes.npy', Y_val)
# se salvan las pruebas
np.save(numpy_dir+'/test_set.npy', X_test)
np.save(numpy_dir+'/test_classes.npy', Y_test)

print("Preprocesamiento Exitoso")
