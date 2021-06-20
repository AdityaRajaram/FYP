import numpy as np
import pandas as pd
import cv2
import os 
from PIL import Image
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten,MaxPool2D
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.layers import Activation, Convolution2D, Dropout, Conv2D,AveragePooling2D, BatchNormalization,Flatten,GlobalAveragePooling2D
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import random
lb_encod  =  LabelEncoder()
cnn_model = None
def renaming(foldernames, train_path):
  for folder in foldernames:
    for count, filename in enumerate(os.listdir(os.path.join(train_path, folder))):
      extension = filename.split('.')[1]
      newFileName = '/' + str(count) + '.' + extension
      os.rename(os.path.join(train_path, folder)+'/'+filename, os.path.join(train_path, folder)+newFileName)
  return True

def ReadImages(foldernames, train_path):
  images  =  []       
  labels  =  [] 
  for folder in foldernames:
    for filename in os.listdir(os.path.join(train_path, folder)):
      if filename.split('.')[1] in ['jpg', 'JPG']:
        img =  cv2.imread(os.path.join(train_path+'/'+folder, filename))
        arr = Image.fromarray(img,'RGB')
        img_arr = arr.resize((50,50))
        labels.append(folder)
        images.append(np.array(img_arr))
  return labels, images

def get_cnn_model():
    l2_reg = 0.001
    opt = Adam(lr = 0.001)
    #Defining the CNN Model
    cnn_model  =  Sequential()
    cnn_model.add(Conv2D(filters = 32, kernel_size = (2,2), input_shape = (50,50, 3), activation = 'relu',kernel_regularizer = l2(l2_reg)))
    cnn_model.add(MaxPool2D(pool_size = (2,2)))
    cnn_model.add(Conv2D(filters = 64, kernel_size = (2,2), activation = 'relu',kernel_regularizer = l2(l2_reg)))
    cnn_model.add(MaxPool2D(pool_size = (2,2)))
    cnn_model.add(Conv2D(filters = 128, kernel_size = (2,2), activation = 'relu',kernel_regularizer = l2(l2_reg)))
    cnn_model.add(MaxPool2D(pool_size = (2,2)))
    cnn_model.add(Conv2D(filters = 64, kernel_size = (2,2), activation = 'relu',kernel_regularizer = l2(l2_reg)))
    cnn_model.add(Dropout(0.1))

    cnn_model.add(Flatten())

    cnn_model.add(Dense(64, activation = 'relu'))
    cnn_model.add(Dense(16, activation = 'relu'))
    cnn_model.add(Dense(7, activation = 'softmax'))

    #CNN Model Summary
    cnn_model.summary()
    cnn_model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
    return cnn_model

def train(train_path):
    foldernames = os.listdir(train_path)
    renaming(foldernames, train_path)
    labels, images = ReadImages(foldernames, train_path)
    np.unique(labels)
    labels = pd.DataFrame(labels)
    labels = lb_encod.fit_transform(labels[0])
    #Saving the image array and corresponding labels
    images = np.array(images)
    np.save("image",images)
    np.save("labels",labels)

    #Loading the images and labels that we have saved above
    image = np.load("image.npy",allow_pickle = True)
    labels = np.load("labels.npy",allow_pickle = True)

    img_shape  = np.arange(image.shape[0])
    np.random.shuffle(img_shape)
    image = image[img_shape]
    labels = labels[img_shape]
    num_classes = len(np.unique(labels))
    len_data = len(image)

    x_train, x_test = image[(int)(0.1*len_data):],image[:(int)(0.1*len_data)]
    y_train,y_test = labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]


    y_train = to_categorical(y_train,num_classes)
    y_test = to_categorical(y_test,num_classes)
    global cnn_model
    cnn_model = get_cnn_model()
    filepath = "weights.hdf5"
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    history = cnn_model.fit(x_train,y_train,batch_size = 128,epochs = 100,verbose = 1,validation_split = 0.33)
    return True

def test(img_path):
    print(img_path)
    img = cv2.imread(img_path)
    arr = Image.fromarray(img,'RGB')
    img_arr = arr.resize((50, 50))
    img_arr = np.expand_dims(img_arr ,axis = 0)
    pred = np.argmax(cnn_model.predict(img_arr),axis = 1)
    prediction  =  lb_encod.inverse_transform(pred)
    return prediction[0]

def test_model(test_path):
    folders_test = os.listdir(test_path)

    renaming(folders_test, test_path)

    t_labels, t_images = ReadImages(folders_test, test_path)

    test_images = np.array(t_images)

    np.save("test_image",test_images)
    test_image = np.load("test_image.npy",allow_pickle = True)

    pred = np.argmax(cnn_model.predict(test_image),axis = 1)
    prediction  =  lb_encod.inverse_transform(pred)

    random_index = random.randint(0, len(test_image))
    test_image = np.expand_dims(test_image[random_index],axis = 0)
    pred_test = np.argmax(cnn_model.predict(test_image),axis = 1)
    prediction_test  =  lb_encod.inverse_transform(pred_test)

    print(prediction_test[0])
    plt.imshow(test_image[0])
    plt.imshow(cv2.cvtColor(test_image[0], cv2.COLOR_BGR2RGB))


