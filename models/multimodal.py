from tensorflow.contrib.keras  import backend as K
from tensorflow.contrib.keras import optimizers, layers, models, callbacks, utils, preprocessing
import glob
import cv2
import numpy as np
from ShatheNet_v2 import *


import time
import random
def randletter():
    CONSONANTS = [chr(i+97) for i in range(26)] + ["", ""]
    VOWELS = ["a", "e", "i", "o", "u"]
    TOTAL = VOWELS + CONSONANTS + VOWELS
    letter = random.choice(TOTAL)
    return letter

def randWord():
    word = []
    for i in range(random.randint(2,8)):
        word += randletter()
    word = "".join(str(x) for x in word) #to convert the list to a      string
    return word #returns a random word



def get_random_description(main_word):
    description = ""
    for i in range(random.randint(2,10)):
        description += randWord() + " "
    if random.random() > 0.5:
        description += str(main_word) + " "
    for i in range(random.randint(2,10)):
        description += randWord() + " "
    return description #returns a random word






#Vectorize the text descriptions for the  network. vectors of 0's and 1's for each character.
#Each description has the size of (max_characters_per_descriptions, num_different_possible_characters)
def vectorize(descriptions_train, descriptions_test):     
    particion_train = len(descriptions_train)   
    descriptions = descriptions_train + descriptions_test
    lens = [len(description) for description in descriptions]
    maxlen_description=max(lens)


    text=' '.join(description for description in descriptions)
    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    sentences = descriptions

    print('Vectorization...')
    X_descriptions = np.zeros((len(sentences), maxlen_description, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X_descriptions[i, t, char_indices[char]] = 1

    X_descriptions_test=X_descriptions[particion_train:]
    X_descriptions_train=X_descriptions[:particion_train]

    return X_descriptions_train, X_descriptions_test

# Returns the data (multimodal Xs and Y from a dataset)
def getData_subfolder(dataFolder='Dataset/train', n_training_samples= 300, n_classes=10):
    read_data = 0
    samples_per_class = n_training_samples / n_classes
    dimension = (192, 192)
    channels = 3
    dimension_images = (n_training_samples,) + dimension + (channels,) 
    dimension_text = (n_training_samples,) + dimension 
    dimension_labels = (n_training_samples)
    images = np.zeros(dimension_images, dtype=np.float16)
    labels = np.zeros(dimension_labels, dtype=np.uint8)
    dic = [None] * 10 #dictionary index -> label
    descriptions  = [None] * n_training_samples # descriptions of each image (multimodal input. Now is simulated)

    index_label = 0 
    #Read the images an create the simulate descriptions
    for folder in glob.glob(dataFolder+ "/*"):
        i = 0
        dic[index_label]= str(folder.split('/')[len(folder.split('/'))-1])
        for image in glob.glob(folder + "/*"):
            descriptions[read_data] = get_random_description(dic[index_label])
            read_data = read_data + 1
            img = cv2.imread(image)
            img = cv2.resize(img, dimension) 
            index_image = samples_per_class*index_label+i
            images[index_image,:,:,:]=img
            labels[index_image] = index_label
            if i + 1 >= samples_per_class:
                break
            else: 
                i = i + 1
        index_label = index_label + 1

    from keras.utils.np_utils import to_categorical
    categorical_labels = to_categorical(labels, num_classes=n_classes)
    #labels_ones[np.arange(dimension_labels), labels] = 1
    if read_data != n_training_samples:
        print('n_training_samples:' + str(n_training_samples))
        print('loaded data:' + str(read_data))


    return images, descriptions, categorical_labels, dic




# Returns the data (multimodal Xs and Y from a dataset)
def getData(dataFolder='Dataset/', n_training_samples_train = 300,n_training_samples_test= 300, n_classes=10):
    images_train, descriptions_train, categorical_labels_train, dic = getData_subfolder(dataFolder=dataFolder + 'train', n_training_samples= n_training_samples_train, n_classes=n_classes)
    images_test, descriptions_test, categorical_labels_test, _ = getData_subfolder(dataFolder=dataFolder + 'test', n_training_samples= n_training_samples_test, n_classes=n_classes)
    X_descriptions_train, X_descriptions_test = vectorize(descriptions_train, descriptions_test)
    return images_train, X_descriptions_train, categorical_labels_train, images_test, X_descriptions_test, categorical_labels_test, dic








def ShatheNet_v2_0_multimodal(n_classes=256, weights=None, shape_images=(192, 192, 3), shape_text=(140, 256)):

    input_image = layers.Input(shape=shape_images)
    input_text = layers.Input(shape=shape_text)
    # x_text = layers.LSTM(128,  input_shape=shape_text)(input_text)
    # if you want more lstm layers, the previous has to have:  return_sequences=True
    x_text = layers.Conv1D(128, 3, padding='valid', activation='relu', input_shape=shape_text)(input_text)
    x_text = layers.GlobalMaxPooling1D()(x_text)
    #aqui podrias meter antes otras densas o LSTM
    x = layers.Conv2D(16, (7, 7), padding='same', activation='relu', input_shape=shape_images,
                     kernel_initializer='truncated_normal', strides=(5, 5))(input_image)
    print (x.shape)

    x = conv2d_bn(x, 16, 5, 5, padding='same', strides=(2, 2))
    x = conv2d_bn(x, 32, 1, 1, padding='same', strides=(2, 2))
    x = conv2d_bn(x, 32, 3, 3, padding='same', strides=(2, 2))
    print (x.shape)

    x = layers.MaxPooling2D((2, 2))(x)
    x = dense_block(x, 3, 32)
    x = layers.GlobalAveragePooling2D()(x)
    concat = layers.concatenate([x, x_text], axis=1)

    x = layers.Dense(n_classes)(concat)
     # la concatenacion opdria ser antes de esta densa (antes de n=clases) ye incluso tener mas densas posteriores
    predictions = layers.Activation('softmax')(x)
    model = models.Model(inputs=[input_image, input_text], outputs=predictions)
    if weights:
        model.load_weights(weights)
    return model





def ShatheNet_text(n_classes=256, weights=None, shape_images=(192, 192, 3), shape_text=(140, 256)):

    input_text = layers.Input(shape=shape_text)
    # x_text = layers.LSTM(128,  input_shape=shape_text)(input_text)
    # if you want more lstm layers, the previous has to have:  return_sequences=True
    x_text = layers.Conv1D(128, 3, padding='valid', activation='relu', input_shape=shape_text)(input_text)
    x_text = layers.GlobalMaxPooling1D()(x_text)

    x = layers.Dense(n_classes)(x_text)
     # la concatenacion opdria ser antes de esta densa (antes de n=clases) ye incluso tener mas densas posteriores
    predictions = layers.Activation('softmax')(x)
    model = models.Model(inputs=input_text, outputs=predictions)
    if weights:
        model.load_weights(weights)
    return model


def ShatheNet_image(n_classes=256, weights=None, shape_images=(192, 192, 3), shape_text=(140, 256)):

    input_image = layers.Input(shape=shape_images)
    # x_text = layers.LSTM(128,  input_shape=shape_text)(input_text)
    # if you want more lstm layers, the previous has to have:  return_sequences=True
    #aqui podrias meter antes otras densas o LSTM
    x = layers.Conv2D(16, (7, 7), padding='same', activation='relu', input_shape=shape_images,
                     kernel_initializer='truncated_normal', strides=(5, 5))(input_image)
    x = conv2d_bn(x, 16, 5, 5, padding='same', strides=(2, 2))
    x = conv2d_bn(x, 32, 1, 1, padding='same', strides=(2, 2))
    x = conv2d_bn(x, 32, 3, 3, padding='same', strides=(2, 2))
    x = layers.MaxPooling2D((2, 2))(x)
    x = dense_block(x, 3, 32)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(n_classes)(x)
     # la concatenacion opdria ser antes de esta densa (antes de n=clases) ye incluso tener mas densas posteriores
    predictions = layers.Activation('softmax')(x)
    model = models.Model(inputs=input_image, outputs=predictions)
    if weights:
        model.load_weights(weights)
    return model




