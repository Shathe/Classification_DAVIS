from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras import optimizers, layers, models, callbacks, utils, preprocessing
import glob
import ntpath
import cv2
import numpy as np
import random
import os
from keras.utils.np_utils import to_categorical
import time

class Loader:
    def __init__(self, Dataset, train_samples, test_samples, N, height=40, width=40, n_classes=10, classes=[]):
        self.N = int(N)
        self.Dataset = Dataset
        self.height = int(height)
        self.width = int(width)
        self.X_train = np.zeros([train_samples, self.height, self.width, self.N], dtype=np.uint8)
        self.X_test = np.zeros([test_samples, self.height, self.width, self.N], dtype=np.uint8)
        self.Y_train = np.zeros([train_samples], dtype=np.uint8)
        self.Y_test = np.zeros([test_samples], dtype=np.uint8)
        self.n_classes = n_classes
        self.classes = classes
        random.seed(os.urandom(9))

    # Gets a sample (combinations of the temporal images of the dataset of the same sample) given an identifier
    # The first index is the first index of the sample. Each sample has several images identified with an index
    def get_sample_from_idenfitier(self, identifier, class_folder, first_index=3):

        sample = np.zeros([self.height, self.width, self.N], dtype=np.uint8)
        # Reads the [N] temporal dimenions
        for index_to_copy in xrange(self.N):
            index_to_read = first_index + index_to_copy

            next_image_to_read = class_folder + '/' + str(identifier) + '_' + str(
                index_to_read) + '.png'

            # Samples og Big datasets dont have indexes
            if 'Big' in self.Dataset:  # Big datasets has no idexes
                next_image_to_read =  class_folder + '/' + str(identifier)  

            # Converts the images to read (3 channels to 1 channel)
            img = cv2.imread(next_image_to_read)
            img = cv2.resize(img, (self.height, self.width), interpolation=cv2.INTER_NEAREST)
            on_events = img[:, :, 2]
            off_events = img[:, :, 0]
            sample[off_events > 0, index_to_copy] = 255
            sample[on_events > 0, index_to_copy] = 127

            return sample

    # Loads all the samples from a [Dataset]. Each sample will have N frames/N-depth.
    def load_data(self, train=True, test=True):
        folders=[]
        if train:
            folders = np.append(folders, 'TRAIN/')
        if test:
            folders = np.append(folders, 'TEST/')

        print(folders)
        for folder in folders:
            # For every folder in the test and train folder, ther's a class
            try:
                samples_seen = []
                for class_folder in glob.glob(self.Dataset + folder + '*'):

                    print(class_folder)
                    class_name = ntpath.basename(class_folder)
                    class_label = self.classes.index(class_name)
                    
                    # These images can be from the same sample
                    # Here the images filenames folow the same rule -> SampleIdentifier_index.png
                    # [N] has to be smaller than the number of images per sample
                    first_index = 3  # in this datasets the first index is 3

                    # Search all the files
                    for file in glob.glob(class_folder + '/*'):

                        filename = ntpath.basename(file)
                        sampleIdentifier = filename.split('_')[0]

                        # If it's the first time to see that identifier, it's the first time to see that sample.
                        # so, load the sample
                        if sampleIdentifier not in samples_seen:
                            # New sample
                            sample = self.get_sample_from_idenfitier(sampleIdentifier, class_folder)
                            if 'TRAIN' in folder:
                                self.X_train[len(samples_seen), :, :, :] = sample
                                self.Y_train[len(samples_seen)] = class_label

                            else:
                                self.X_test[len(samples_seen), :, :, :] = sample
                                self.Y_test[len(samples_seen)] = class_label

                            samples_seen = np.append(samples_seen, sampleIdentifier)
            except Exception as e:
                print(e)
                
        self.Y_test = to_categorical(self.Y_test, num_classes=len(self.classes))
        self.Y_train = to_categorical(self.Y_train, num_classes=len(self.classes))




    # Returns a random batch
    def get_batch(self, size=32, train=True):
        first = time.time()
        folder_to_look = 'TEST/'
        if train:
            folder_to_look = 'TRAIN/'

        x = np.zeros([size, self.height, self.width, self.N], dtype=np.uint8)
        y = np.zeros([size], dtype=np.uint8)
        for index in xrange(size):
            # Aleatoriamente elegir una clase.
            class_folder = random.choice(os.listdir(self.Dataset + folder_to_look))


            class_name = ntpath.basename(class_folder)
            class_label = self.classes.index(class_name)

            # Aleatoriamente elegir un fichero (y cargar las N dimensiones
            file = random.choice(os.listdir(self.Dataset + folder_to_look +  class_folder))

            filename = ntpath.basename(file)
            sampleIdentifier = filename.split('_')[0]
            sample = self.get_sample_from_idenfitier(sampleIdentifier, self.Dataset + folder_to_look + class_folder)


            x[index, :, :, :] = sample
            y[index] = class_label


        y = to_categorical(y, num_classes=self.n_classes)
        second = time.time()
        #print(str(second - first) + " seconds to load")
        return x, y



if __name__ == "__main__":
    loader = Loader(Dataset='Datasets/MNIST-Big/', train_samples=0, test_samples=0, n_classes=10, classes=['3', '8', '6', '4', '0', '5', '2', '9', '1', '7'],
                N=1, height=40, width=40)

    loader.get_batch(size=64)
