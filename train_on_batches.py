import argparse
from tensorflow.contrib.keras import optimizers, layers, models, callbacks, utils, preprocessing, metrics, applications
import tensorflow.contrib.keras
import numpy as np
import glob
import os
import random
from shutil import copyfile
import cv2

from Loader import Loader
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Dataset to train") # 'Datasets/MNIST-Big/'
parser.add_argument("--dimensions", help="Temporal dimensions to get from each sample")
args = parser.parse_args()


epochs = 250
learning_rate = 0.001
n_batches = 32
n_classes = 0
train_samples = 60000
test_samples = 10000

for _, dirnames, filenames in os.walk(args.dataset + 'TRAIN'):
  # ^ this idiom means "we won't be using this value"
    n_classes += len(dirnames)
    break

loader = Loader(Dataset=args.dataset, train_samples=train_samples, test_samples=test_samples, n_classes=n_classes,
                N=args.dimensions, height=192, width=192)





# create the base pre-trained model
base_model = applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_shape=(192, 192, 3))
# print(base_model.load_weights("my_model_final.h5", by_name=True))

predictions = layers.Dense(n_classes, activation='softmax')(base_model.output)

# this is the model we will train
model = models.Model(inputs=base_model.input, outputs=predictions)

model.summary()

adam = optimizers.Adam(lr=learning_rate) # decay=0.0001? decay 1/(1+decay*epochs*batches_per_epoch)*lr
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.9, patience=2, min_lr=0.00001)

for batches_train in xrange(epochs * (train_samples // n_batches)):
    x, y = loader.get_batch(size=n_batches, train=True, n_classes = 10)
    print(model.train_on_batch(x, y))


#faltaria cargar todas las Y y evaluar


'''
model.fit(loader.X_train, loader.Y_train, batch_size=n_batches, epochs=epochs, validation_data=(loader.X_test, loader.Y_test), callbacks=[reduce_lr])

score = model.evaluate(loader.X_test, loader.Y_test, batch_size=n_batches)
#model.save('my_model_final.h5')

print('Test loss:', score[0])
print('Test accuracy:', score[1])

'''