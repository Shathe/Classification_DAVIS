import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
from Layers import *
from DenseNet import *
import random
import os
import argparse
from Loader import Loader
import cv2
random.seed(os.urandom(9))
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# python train_tensorflow --dataset Datasets/MNIST-Normal/ --dimensions 3
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Dataset to train") # 'Datasets/MNIST-Big/'
parser.add_argument("--dimensions", help="Temporal dimensions to get from each sample")
args = parser.parse_args()


# Hyperparameter
init_learning_rate = 5e-4
dropout_rate = 0.2

# Label & batch_size
batch_size = 32

total_epochs = 50
training_samples = 60000 

width = 40
height = 40
channels = int(args.dimensions)
classes= os.listdir(args.dataset + 'TRAIN/' )
n_classes = sum((os.path.isdir(args.dataset + 'TRAIN/' + i) for i in classes))
print(str(n_classes) + ' clases a entrenar')


loader = Loader(Dataset=args.dataset, train_samples=0, test_samples=0, n_classes=n_classes, classes=classes,
                N=args.dimensions, height=height, width=width)



#Necesario apra algunas operaciones como dropouts que funcionan diferente dependiendo de si es training o testing
training_flag = tf.placeholder(tf.bool)

#Placeholder para las imagenes. 
x = tf.placeholder(tf.float32, shape=[None, height, width, channels])
batch_images = tf.reshape(x, [-1, height, width, channels])


#Placeholders para las clases (vector de salida que seran valores de 0-1 por cada clase)
label = tf.placeholder(tf.float32, shape=[None, n_classes])


#Para poder modificarlo
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

output = DenseNet(x=batch_images, nb_blocks=2, filters=12, n_classes=n_classes, training=training_flag).model

#funcion de coste: cross entropy (se pued modificar. mediado por todos los ejemplos)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output))

# Uso el optimizador de Adam y se quiere minimizar la funcion de coste
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon= 1e-8)
train = optimizer.minimize(cost)

#Accuracy es:
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        batch_x_test, batch_y_test = loader.get_batch(size=1, train=False)
        batch_x_test = batch_x_test.astype(np.float16)/255 - 0.5

        cv2.imshow('image',batch_x_test[0,:,:,:].astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        predictions = sess.run(output, feed_dict={x: batch_x_test, training_flag : False})
        predicted_class = np.argmax(predictions)

        print('Clase real =', loader.classes[predicted_class]) 
    else:
        print("No se encuentran los pesos")

    