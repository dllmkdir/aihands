# carga de librerías
from sklearn.metrics import accuracy_score
from keras.applications.vgg16 import preprocess_input
from keras.applications import VGG16
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# se abre el modelo y los pesos guardados en el JSON y H5 generados en el entrenamiento
with open('Model/model.json', 'r') as f:
    model = model_from_json(f.read())
    model.summary()
model.load_weights('Model/model_weights.h5')

# se abren las clases para realizar pruebas
X_test = np.load("Numpy/test_set.npy")
Y_test = np.load("Numpy/test_classes.npy")

# Se hace la predicción del modelo utilizando solo las imágenes de pruebas
Y_predict = model.predict(X_test)

# se extra las clasificaciones obtenidas por el modelo
# recordatorio: A:0, B:1, C:2
print("Predicción exitosa")
Y_predict = [np.argmax(r) for r in Y_predict]
Y_test = [np.argmax(r) for r in Y_test]

# se muestra que tan acertados fue la predicción
acc_score = accuracy_score(Y_test, Y_predict)
print("Accuracy: "+str(acc_score))
