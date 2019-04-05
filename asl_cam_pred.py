from keras.applications.vgg16 import preprocess_input
from sklearn.metrics import accuracy_score
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications import VGG16
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# se abre el modelo y los pesos guardados en el JSON y H5 generados en el entrenamiento
with open('Model/model.json', 'r') as f:
    model = model_from_json(f.read())
    model.summary()
model.load_weights('Model/model_weights.h5')


scaling_factorx = 1.5
scaling_factory = 1.5
image_size = 224

# captura de video
cap = cv2.VideoCapture(0)

# terminar cuando se detecte una interrupción de teclado
while (cv2.waitKey(10) & 0xff) != 27:
    rec_obj, frame = cap.read()  # lectura de cámara
    box = cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 5)  # crear rectángulo verde de captura
    frame = cv2.resize(frame, None, fx=scaling_factorx, fy=scaling_factory, interpolation=cv2.INTER_AREA)  # escalamiento de imagen
    frame = cv2.flip(frame, 1)  # espejeo de imagen

    # muestra de la imagen
    cv2.imshow('video output', frame)

    # escalamient de la imagen y captura del recuadro
    frame = cv2.resize(frame, (image_size, image_size))
    img_data = image.img_to_array(frame)
    img_data = np.expand_dims(img_data, axis=0)
    # preprocesamiento para vgg16
    img_data = preprocess_input(img_data)
    # predicción del modelo
    vgg16_category = model.predict(img_data)
    category = np.argmax(vgg16_category, axis=1)
    print(category)  # mostrar categoría resultante
# liberación de recursos
cap.release()
cv2.destroyAllWindows()
