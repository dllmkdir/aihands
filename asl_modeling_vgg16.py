# carga de keras, numpy,etc
import numpy as np
from keras.optimizers import SGD
from keras.applications.vgg16 import preprocess_input
from keras.applications import VGG16
from keras.preprocessing import image
from keras.models import model_from_json
from keras import models, layers, optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge
from keras.layers import Input, Dense, Reshape, Activation

# Modelo VGG16 es cargado
# la resolución de la imagen en preprocesamiento fue de 224x224x3 (RGB)
# include_top=False=>evita que se cargue la última capa de perceptrones
image_size = 224
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

# El modelo se escribirá de manera secuencial
model = models.Sequential()

# se agrega la base de 13 capas de convolución y 5 maxpooling al modelo
model.add(vgg_base)

# últimas imágenes son aplanadas para entrar a la capa de perceptrones
model.add(layers.Flatten())

# primera capa: 8192 neuronas de activación relu
# Relu: 6 veces más rápido que otras activaciones, implementación sencilla
model.add(Dense(8192, activation='relu'))
model.add(Dropout(0.8))
# segunda capa: 4096 neuronas de activación relu
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))  # prevent overfitting
# capa final de clasificación: 3 neuronas con activación softmax para normalización
model.add(Dense(3, activation='softmax'))  # last layer
# softmax => 0,1
# A =>0.3
# b =>0.4
# c =>0.3

# optimización y creación de checkpoints
sgd = SGD(lr=0.001)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint("Checkpoint/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

# obtención de datos de entrenamiento
X_train = np.load("Numpy/train_set.npy")
Y_train = np.load("Numpy/train_classes.npy")

# obtención de datos de validación
X_valid = np.load("Numpy/validation_set.npy")
Y_valid = np.load("Numpy/validation_classes.npy")


# entrenamiento del modelo completo en 3 épocas
model.fit(X_train/255.0, Y_train, epochs=3, batch_size=32, validation_data=(X_valid/255.0, Y_valid), shuffle=True, callbacks=[checkpoint])

# modelo y pesos son guardados en JSOn y H5
model_json = model.to_json()
with open("Model/model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("Model/model_weights.h5")
print("El Entrenamiento ha finalizado")
