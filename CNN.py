import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

#initializing CNN
classifier_cnn=Sequential()

#adding hidden layer (convolution)
classifier_cnn.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
# a,b,c a= number of feature detector for a single image(32)
#b=number of rows in single feature detector
#c=number of coloumns in a single feature detector, border
#input shae=abc, a=3for 3 d maarray, 256,256 is for pixles


#pooling (for reducing the size of the image)
classifier_cnn.add(MaxPooling2D(pool_size=(2,2)))

classifier_cnn.add(Convolution2D(32,3,3,activation='relu'))#adding for accuracy, down comment,can add 3rd layer with 62 insted 32
classifier_cnn.add(MaxPooling2D(pool_size=(2,2)))#same as above line


#Flattening (craete a single vector for Ann)
classifier_cnn.add(Flatten())

#fulll connection (Ann)
classifier_cnn.add(Dense(units=128,activation='relu'))
#units=number nodes in the hidden layer,common to choose power of 2 and near 100
classifier_cnn.add(Dense(units=1,activation='sigmoid'))

#compilling the Cnn
classifier_cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#loss=categrical_crosscentropy if more than 2 dependent variable

#fitting the CNN to our images dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
        rescale=1./255, #pixel value will be between 0and 255
        shear_range=0.2,#random trasaction
        zoom_range=0.2,# random zooms
        horizontal_flip=True) #flipped horizontally

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('dataset/training_set',target_size=(64,64),batch_size=32,class_mode='binary')
#traget size is size expected in your cnn model
#we can increase the size (64,64) to have more features in the model
test_set = test_datagen.flow_from_directory('dataset/test_set',target_size=(64,64),batch_size=32,class_mode='binary')

# fine-tune the model
classifier_cnn.fit(train_set,steps_per_epoch=8000,epochs=25,validation_data=test_set,validation_steps=2000)


#increase the accuracy
#1-add another convolutional layer
#2-add full connection ann layer


#single prediction
from keras.preprocessing import image

test_image=image.load_img('dataset/single_prediction/Photo (1) (1).jpeg',target_size=(64,64))
test_image=image.img_to_array(test_image)#to make it in 3d array as we have done above in step 1
test_image=np.expand_dims(test_image,axis=0)
test_image = test_image / 255.0
result=classifier_cnn.predict(test_image)
train_set.class_indices
if result[0][0]>=0.6:
    prediction='Dog'
else:
    prediction='Cat'
        

