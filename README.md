# Machine Learning y el LSM
LSM(Lengua de señas mexicana) es la lengua de señas dominante en México. Sin embargo es poco común que los ciudadanos de las regiones de México tengan dominio de esta lengua si no tienen necesidad de usarla como su forma primaria de comunicación o están directamente relacionados con alguien que la use. Sin embargo en un mundo globalizado el poder comunicarse entre hablantes de diferentes lenguas a través de la tecnología se ha convertido en algo prioritario.

Tecnologías: keras y tensorflow

# Procedimiento

Primero se hizo un preprocesamiento en el que se redimensionan y separan las imágenes correspondientes de cada letra en tres secciones principales: entrenamiento, validación y prueba.
Posteriormente se hizo el entrenamiento empleando Keras VGG16 para la red neuronal, a su vez se agregan capas de neuronas con activación de tipo ReLU y por último una capa de neuronas de activación softmax.
Ya teniendo la red se entrenó empleando las imágenes que fueron separadas en el preprocesamiento y a partir de pruebas se determina el accuracy de la red ya entrenada. Por último como prueba final se emplea la red entrenada para clasificar en tiempo real las señas de la mano de un individuo, para las que la red fue entrenada.


Para lograr resolver el problema planteado para leer el lenguaje de señas, se utilizarán redes neuronales que a través de su entrenamiento proveerán la solución esperada. El campo que engloba a las redes neuronales es de vital importancia para la tecnología contemporánea, ya que muchos avances y descubrimientos se están logrando en esta área cada año. Utilizar redes neuronales para la clasificación de los gestos propuestos es de hecho, una implementación que conlleva al entrenamiento de una red con los gestos manuales (lenguaje de señas). Para esta tarea de clasificación de imágenes,  será utilizada una red neuronal convolucional. Aunado a esto, las redes neuronales convolucionales poseen distintas características que las hacen una opción perfecta para la lectura de imágenes. A pesar de esto, no todas las CNNs logran el mismo propósito siempre. Debido a que existen diferentes distribuciones para aplicar las operaciones correspondientes a este tipo de red (típicamente convoluciones y pooling), se optará por utilizar la arquitectura VGG16. Esta arquitectura de CNN cuenta con una larga aceptación, al igual que la arquitectura de AlexNet. VGG16 cuenta con 13 capas de convolución y 5 de maxpooling distribuidas en 5 etapas. A continuación se puede observar cómo se conforma esta arquitectura:

![](https://qph.fs.quoracdn.net/main-qimg-e657c195fc2696c7d5fc0b1e3682fde6)

Observe que después del quinto maxpooling se procede a aplanar la información restante y lograr el procesamiento restante a través de una red de perceptrón multicapa para generar la clasificación final. VGG16 es galardonado por múltiples concursos de clasificación de imágenes que se han hecho en los últimos años, incluso su variante VGG19 ha superado implementaciones de redes pasadas. A pesar de esto la solución también se extiende a mejorar el tiempo de convergencia del modelo. Este modelo puede ser entrenado de cero para lograr la clasificación; Sin embargo, se optó por utilizar una técnica para la convergencia rápida. Esta técnica es conocida como transfer learning, y es utilizada para entrenar una red neuronal a través del conocimiento previo adquirido de otra implementación. En términos sucintos, se refiere a entrenar una red que ya cuenta con pesos que fueron encontrados para una aplicación más general. Con la técnica de  transfer learning,  el modelo logra la clasificación deseada en un par de horas,  mientras que comparándolo con entrenamiento de cero podría llevar varios días. La red neuronal será entrenada utilizando la base de datos de American Sign Language que provee Kaggle.com. Esta base cuenta con 3000 imágenes por letra con una resolución de 200x200 pixeles en RGB. Para descargar la biblioteca de imágenes se puede acceder a la siguiente liga:
https://www.kaggle.com/grassknoted/asl-alphabet/version/1

La versión americana del lenguaje de señas cuenta con simbología parecida a la mexicana, por lo que no se descartó utilizar las señas de Kaggle que son idénticas para la implementación final.


Para la implementación de la red neuronal se decidió empezar con la clasificación de tres letras para conocer los alcances del proyecto. Las letras que se decidieron clasificar son “A”, “B” y “C”. A continuación se puede observar se equivalente en lenguaje de señas:

![](https://i.pinimg.com/originals/f3/83/79/f38379978ac49a60af3d4e498c6937ba.gif)

# Resultados

Con este procedimiento se logró el debido entrenamiento de la red neuronal convolucional.  Keras provee el parámetro ‘accuracy’ en métricas para conocer lo exitosa que está siendo la red. Utilizar la técnica de  transfer learning  resultó en un rotundo éxito en el entrenamiento, ya que con tan solo 3 épocas el modelo logró un “accuracy” de 99.2% con una pérdida del 2.5%.

En la etapa de pruebas, se encontró que todas las imágenes de pruebas fueron clasificadas correctamente, teniendo un 100% de exactitud.


Finalmente el programa fue puesto a prueba utilizando la cámara de la computadora para observar la clasificación. Los tres gestos fueron reconocidos apropiadamente, sin embargo existían varias ocasiones en que la red no podía leer bien el gesto que se estaba representando. Viendo estas diferencias de resultados, pudimos determinar que la red neuronal sufrió de “overfitting” con los datos de ASL de Kaggle.com. Este problema puede ser erradicado con algunas técnicas para mejorar el modelo. A pesar de esto se pudieron capturar varios momentos en que la red funcionó correctamente. En resumen la red neuronal tuvo un desempeño “bueno”, más no “excelente” como se esperaba tres la correcta clasificación de las 750 imágenes de prueba.
