from keras.utils import np_utils
from random import shuffle
import numpy as np
import random
import cv2
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class_dic = {"A": 0, "B": 1, "C": 2}

image_list = []
image_class = []

path = "asl_data/new"

data_folder_path = "asl_data/new"

# alphabet directory
alphabet_dir = os.listdir(data_folder_path)


print("Shuffled the Dataset!")

class_count = {'A': 0, 'B': 0, 'C': 0}

X = []
Y = []
X_val = []
Y_val = []
X_test = []
Y_test = []
unique_list = []
for alpha_dir in alphabet_dir:
    temp_alphabet_dir = os.path.join(ROOT_DIR, data_folder_path, alpha_dir)
    print(temp_alphabet_dir)
    for root, dirs, files in os.walk(temp_alphabet_dir):
        for file_name in files:
            #  print(file_name)
            label = file_name[0]
            if label not in unique_list:
                print(label)
                unique_list.append(label)
            path = os.path.join(temp_alphabet_dir, file_name)
            image = cv2.imread(path)
            resized_image = cv2.resize(image, (224, 224))
            if class_count[label] < 2000:
                class_count[label] += 1
                X.append(resized_image)
                Y.append(class_dic[label])
            elif class_count[label] >= 2000 and class_count[label] < 2750:
                class_count[label] += 1
                X_val.append(resized_image)
                Y_val.append(class_dic[label])
            else:
                X_test.append(resized_image)
                Y_test.append(class_dic[label])

print(len(unique_list))

Y = np_utils.to_categorical(Y)
Y_val = np_utils.to_categorical(Y_val)
Y_test = np_utils.to_categorical(Y_test)

print(len(Y))
print(len(Y_val))
print(len(Y_test))

print(len(X))
print(len(X_val))
print(len(X_test))

npy_data_path = "Numpy"

np.save(npy_data_path+'/train_set.npy', X)
np.save(npy_data_path+'/train_classes.npy', Y)

np.save(npy_data_path+'/validation_set.npy', X_val)
np.save(npy_data_path+'/validation_classes.npy', Y_val)

np.save(npy_data_path+'/test_set.npy', X_test)
np.save(npy_data_path+'/test_classes.npy', Y_test)

print("Data pre-processing Success!")
