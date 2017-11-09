from tensorflow.contrib.keras  import backend as K

from tensorflow.contrib.keras import optimizers, layers, models, callbacks, utils, preprocessing
import glob
import cv2
import numpy as np


def ShatheNet_v2(n_classes=256, weights=None):
    # paddign same, filtros mas pequemos.. 
    input_shape = (192, 192, 3)

    inputs = layers.Input(shape=input_shape)

    # a layer instance is callable on a tensor, and returns a tensor
    x = conv2d_bn(inputs, 32, 3, 3, padding='valid', strides=(2, 2))
    x = conv2d_bn(x, 64, 1, 1, padding='valid', strides=(1, 1))
    x = conv2d_bn(x, 64, 3, 3, padding='valid', strides=(1, 1))
    x = layers.MaxPooling2D((2, 2))(x)
    x = dense_block(x, 8, 32)
    x = transition_block(x, 96)
    x = dense_block(x, 12, 32)
    x = transition_block(x, 128)
    x = dense_block(x, 20, 32) 
    x = transition_block(x, 196)
    x = dense_block(x, 16, 32)
    x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Flatten()(x)
    predictions = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=predictions)

    if weights:
        model.load_weights(weights)
    return model



def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1)):
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3

    x = layers.Conv2D(filters, (num_row, num_col), strides=strides, padding=padding)(x) # use_bias=False,
    x = layers.BatchNormalization(axis=bn_axis)(x) # scale=False,
    x = layers.Activation('relu')(x)
    return x

def node(x, nb_filter):

    tower_1 = conv2d_bn(x, nb_filter, 1, 1, padding='same', strides=(1, 1))
    tower_1 = conv2d_bn(tower_1, nb_filter, 3, 3, padding='same', strides=(1, 1))

    tower_2 = conv2d_bn(x, nb_filter, 1, 1, padding='same', strides=(1, 1))
    tower_2 = conv2d_bn(tower_2, nb_filter, 5, 5, padding='same', strides=(1, 1))

    output = layers.concatenate([tower_1, tower_2], axis=3)
    return output
    #ahora toca un denseblock con los nodos


def dense_block(x, nb_layers, nb_filter):
    #Hacer algo como contaenation imapres o pares para ver si se reduce el numero mucho?
    filter_augmenation_step = 4
    concatetation_of_inputs = x
    for i in range(nb_layers):
        next_node = node(concatetation_of_inputs, nb_filter)
        concatetation_of_inputs = layers.concatenate([concatetation_of_inputs, next_node], axis=3)
        previous_node = next_node
        nb_filter = nb_filter + filter_augmenation_step


    return concatetation_of_inputs #hacia la transition layer

def transition_block(x, nb_filter):

    x = conv2d_bn(x, nb_filter, 1, 1, padding='same', strides=(1, 1))
    x = layers.AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x
