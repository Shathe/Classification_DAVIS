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
from imgaug import augmenters as iaa
import imgaug as ia

random.seed(os.urandom(9))

# python train_tensorflow --dataset Datasets/MNIST-Normal/ --dimensions 3
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Dataset to train")  # 'Datasets/MNIST-Big/'
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
classes = os.listdir(args.dataset + 'TRAIN/')
n_classes = sum((os.path.isdir(args.dataset + 'TRAIN/' + i) for i in classes))
print(str(n_classes) + ' clases a entrenar')

loader = Loader(Dataset=args.dataset, train_samples=0, test_samples=0, n_classes=n_classes, classes=classes,
                N=args.dimensions, height=height, width=width)

'''
#Augmentation. (Yet to edit with Chema's code)los valores aleatorios no sirven porque solo se ejecutan una vez
def augmentation(x,  height=height, width=width, training=True, flips=True, rotate_angle=90, color_augmentation=False, shift_x=0, shift_y=0, ratio_augmentation=1):
	if training:
		if random.random() < ratio_augmentation:
			if flips:
				x = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), x)
				x = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), x)

			x = tf.contrib.image.rotate(x, random.randint(-rotate_angle, rotate_angle) , interpolation= "BILINEAR")
			if color_augmentation:
				x = tf.image.random_saturation( x, 0, 0.2, seed=9 )
				x = tf.image.random_hue( x, 0.2, seed=9 )  
				x = tf.image.random_contrast( x, 0, 0.2, seed=9 )
				x = tf.image.random_brightness( x, 0.2, seed=9 )
			tx = random.randint(-shift_x, shift_x)
			ty = random.randint(-shift_y, shift_y)
			transforms = [1, 0, tx, 0, 1, ty, 0, 0]
			x = tf.contrib.image.transform(x, transforms, interpolation="BILINEAR")

	x = tf.image.resize_image_with_crop_or_pad(x, height, width)

	return x
'''

# Necesario apra algunas operaciones como dropouts que funcionan diferente dependiendo de si es training o testing
training_flag = tf.placeholder(tf.bool)
# Placeholder para las imagenes.
x = tf.placeholder(tf.float32, shape=[None, height, width, channels])
batch_images = tf.reshape(x, [-1, height, width, channels])

# Placeholders para las clases (vector de salida que seran valores de 0-1 por cada clase)
label = tf.placeholder(tf.float32, shape=[None, n_classes])

# Para poder modificarlo
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

output = DenseNet(x=batch_images, nb_blocks=2, filters=12, n_classes=n_classes, training=training_flag).model

# funcion de coste: cross entropy (se pued modificar. mediado por todos los ejemplos)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output))

# Uso el optimizador de Adam y se quiere minimizar la funcion de coste
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-8)
train = optimizer.minimize(cost)

# Accuracy es:
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('learning_rate', learning_rate)
tf.summary.image('input', batch_images)

'''
for op in tf.get_default_graph().get_operations():
    try:
        op_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, op.name)
        if op_var != []:
            #print(op_var)
            tf.summary.histogram(op.name, op_var)

        else:
            pass

    except:
        pass 
'''



for op in tf.get_default_graph().get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    #tf.summary.histogram(op.name, op)
    pass


saver = tf.train.Saver(tf.global_variables())

alot = lambda aug: iaa.Sometimes(0.80, aug)
sometimes = lambda aug: iaa.Sometimes(0.50, aug)
few = lambda aug: iaa.Sometimes(0.20, aug)
seq_rgb = iaa.Sequential([

    iaa.Fliplr(0.25),  # horizontally flip 50% of the images
    iaa.Flipud(0.25),  # horizontally flip 50% of the images
    sometimes(iaa.Add((-30, 30))),
    sometimes(iaa.Multiply((0.80,1.20), per_channel=False)),
    sometimes(iaa.GaussianBlur(sigma=(0, 0.20))),
    sometimes(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.00, 0.20), per_channel=0.5)),
    sometimes(iaa.ContrastNormalization((0.7,1.4))),
    sometimes(iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        # scale images to 80-120% of their size, individually per axis
        translate_percent={"x": (-0.20, 0.2), "y": (-0.2, 0.2)},
        # translate by -20 to +20 percent (per axis)
        rotate=(-45, 45),  # rotate by -45 to +45 degrees
        order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
        cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
        mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
    ))])

seq_multi = iaa.Sequential([

    # iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    # iaa.Flipud(0.5),  # horizontally flip 50% of the images
    #sometimes(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.00, 0.20), per_channel=0.5)),

    sometimes(iaa.Affine(
        # scale images to 80-120% of their size, individually per axis
        # translate by -20 to +20 percent (per axis)
        rotate=(-45, 45),  # rotate by -45 to +45 degrees
    ))
])
'''
sometimes(iaa.Affine(
    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    # scale images to 80-120% of their size, individually per axis
    translate_percent={"x": (-0.20, 0.2), "y": (-0.2, 0.2)},
    # translate by -20 to +20 percent (per axis)
    rotate=(-45, 45),  # rotate by -45 to +45 degrees
    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
    mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
))
'''






with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    writer_train = tf.summary.FileWriter('./logs/train', sess.graph)
    writer_test = tf.summary.FileWriter('./logs/test', sess.graph)

    global_step = 0
    epoch_learning_rate = init_learning_rate

    # EPOCHS
    for epoch in range(total_epochs):
        # Simple learning rate decay
        if epoch == (total_epochs * 0.35) or epoch == (total_epochs * 0.65) or epoch == (total_epochs * 0.85):
            epoch_learning_rate = epoch_learning_rate / 10

        total_batch = int(training_samples / batch_size)

        # steps in every epoch
        for step in range(total_batch):
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x, batch_y = loader.get_batch(size=batch_size, train=True)


            images_aug = seq_multi.augment_images(batch_x)

            train_feed_dict = {
                x: batch_x,
                label: batch_y,
                learning_rate: epoch_learning_rate,
                training_flag: True
            }




            _, loss = sess.run([train, cost], feed_dict=train_feed_dict)

            if step % 100 == 0:
                global_step += 100
                train_summary, train_accuracy = sess.run([merged, accuracy], feed_dict=train_feed_dict)
                # accuracy.eval(feed_dict=feed_dict)
                print("Step:", step, "Loss:", loss, "Training accuracy:", train_accuracy)
                writer_train.add_summary(train_summary, global_step=epoch)

        batch_x_test, batch_y_test = loader.get_batch(size=batch_size, train=False)
        test_feed_dict = {
            x: batch_x_test,
            label: batch_y_test,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        test_summary, accuracy_rates = sess.run([merged, accuracy], feed_dict=test_feed_dict)
        writer_test.add_summary(test_summary, global_step=epoch)

        print('Epoch:', '%04d' % (epoch + 1), '/ Accuracy =', accuracy_rates)
        saver.save(sess=sess, save_path='./model/dense.ckpt')
    # writer.add_summary(test_summary, global_step=epoch)

    saver.save(sess=sess, save_path='./model/dense.ckpt')
