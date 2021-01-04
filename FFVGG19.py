
from imutils import paths
import numpy as np
import os
import warnings
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy
from keras.utils import plot_model
import tensorflow as tf

in_shape = (300, 300, 3)
classes = 2


def make_lvgg (in_shape, tune, classes):
    
    
    base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)

    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])

    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[20:]:
        layer.trainable = True
    
    early2 = layer_dict['block2_pool'].output 
    #early2 = Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early2)
    early2 = tf.keras.layers.BatchNormalization()(early2)
    early2 = tf.keras.layers.Dropout(0.5)(early2)
    early2= tf.keras.layers.GlobalAveragePooling2D()(early2)
        
    early3 = layer_dict['block3_pool'].output   
    #early3 = Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early3)
    early3 = tf.keras.layers.BatchNormalization()(early3)
    early3 = tf.keras.layers.Dropout(0.5)(early3)
    early3= tf.keras.layers.GlobalAveragePooling2D()(early3)    
        
    early4 = layer_dict['block4_pool'].output   
    #early4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early4)
    early4 = tf.keras.layers.BatchNormalization()(early4)
    early4 = tf.keras.layers.Dropout(0.5)(early4)
    early4= tf.keras.layers.GlobalAveragePooling2D()(early4)     
    
    x1 = layer_dict['block5_conv3'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1)
    x = tf.keras.layers.concatenate([x1, early4, early3], axis=-1)  
    
    #x = Flatten()(x)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(2500, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model

