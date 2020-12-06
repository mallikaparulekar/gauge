
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import utils
import os
#%matplotlib inline

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

from IPython.display import SVG, Image

#can also try from livelossplot import PlotLossesKerasTF
#from livelossplot import PlotLossesKerasTF
from livelossplot.tf_keras import PlotLossesCallback # from web

#from livelossplot.keras import PlotLossesCallback
import tensorflow as tf
print("Tensorflow version:", tf.__version__)


#displays a few of the training images
#utils.datasets.fer.plot_example_images(plt).show()


#prints the number of images we have for each emotion
for expression in os.listdir("train/"):
    if expression != ".DS_Store":
        print(str(len(os.listdir("train/" + expression))) + " " + expression + " images")


#CAN MAKE MORE GENERAL--this is only for 48 by 48 images
img_size = 48
#what does batch size mean?--can mess around with
batch_size = 64
#mess around with horizontal flip= true, add more params under imageDataGenerator (data augmentation?)
datagen_train = ImageDataGenerator(horizontal_flip=True)
train_generator = datagen_train.flow_from_directory("train/", target_size = (img_size, img_size), color_mode = "grayscale", batch_size=batch_size, class_mode = "categorical",shuffle = True )

datagen_validation = ImageDataGenerator(horizontal_flip=True)
validation_generator = datagen_train.flow_from_directory("test/", target_size = (img_size, img_size), color_mode = "grayscale", batch_size=batch_size, class_mode = "categorical",shuffle = False)


#USING ORIGINAL CNN ARCHITECTURE

model = Sequential()

#1 - conv
model.add(Conv2D(64, (3, 3), padding = 'same', input_shape=(48, 48, 1))) #64 filters, each 3 x 3 filters, padding is same
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25)) # prevents overfitting. Can change '0.25' value

#2 - conv layer
model.add(Conv2D(128, (5, 5), padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25)) # prevents overfitting. Can change '0.25' value

#3 - conv layer
model.add(Conv2D(512, (3, 3), padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25)) # prevents overfitting. Can change '0.25' value

#4 - conv layer
model.add(Conv2D(512, (3, 3), padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25)) # prevents overfitting. Can change '0.25' value

model.add(Flatten())

#1st fully connected layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25)) # prevents overfitting. Can change '0.25' value

#last fully connected layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25)) # prevents overfitting. Can change '0.25' value

model.add(Dense(7, activation='softmax'))

opt = Adam(lr = 0.0005)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#Training and Evalutaing Model
epochs = 10
steps_per_epoch = train_generator.n//train_generator.batch_size
validation_steps = validation_generator.n//validation_generator.batch_size
checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy',
                             save_weights_only=True, mode = 'max', verbose = 1) # saves the weight of most accurate model

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience = 2, min_lr=0.00001, model='auto') # reduces learning rate when there is a plateau in validation

callbacks = [PlotLossesCallback(), checkpoint, reduce_lr]
#callbacks=[PlotLossesKerasTF(), checkpoint, reduce_lr]

history = model.fit(
    x=train_generator,
    steps_per_epoch = steps_per_epoch,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = validation_steps,
    #callbacks = callbacks

)

#Representing Model as JSON string

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)





