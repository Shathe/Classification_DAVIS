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
from augmenters import get_augmenter

random.seed(os.urandom(9))

# tensorboard --logdir=train:./gs/train,test:./logs/test/
# python train.py --dataset ./MNIST-Normal/ --dimensions 3 --augmentation True --tensorboard True

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Dataset to train")  # 'Datasets/MNIST-Big/'
parser.add_argument("--dimensions", help="Temporal dimensions to get from each sample")
parser.add_argument("--tensorboard", help="Monitor with Tensorboard", default=False)
parser.add_argument("--augmentation", help="Image augmentation", default=False)
parser.add_argument("--init_lr", help="Initial learning rate", default=5e-4)
parser.add_argument("--batch_size", help="batch_size", default=32)
parser.add_argument("--epochs", help="Number of epochs to train", default=100)
parser.add_argument("--width", help="width", default=40)
parser.add_argument("--height", help="height", default=40)
parser.add_argument("--dropout_rate", help="dropout_rate", default=0.2)
args = parser.parse_args()

# Hyperparameter
init_learning_rate = float(args.init_lr)
dropout_rate = float(args.dropout_rate)
augmentation = args.augmentation
tensorboard = args.tensorboard
batch_size = int(args.batch_size)
total_epochs = int(args.epochs)
width = int(args.width)
height = int(args.height)
channels = int(args.dimensions)



training_samples = 60000
test_samples = 10000
classes = os.listdir(args.dataset + 'TRAIN/')
n_classes = sum((os.path.isdir(args.dataset + 'TRAIN/' + i) for i in classes))
print(str(n_classes) + ' Classes to train')

loader = Loader(Dataset=args.dataset, train_samples=training_samples, test_samples=test_samples, n_classes=n_classes, classes=classes,
                N=args.dimensions, height=height, width=width, batches_loading=True)



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


if tensorboard:
    # Scalar summaries
    tf.summary.scalar('loss', cost)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('learning_rate', learning_rate)

    #Input summary
    if args.dimensions == 3:
        tf.summary.image('input', batch_images, max_outputs=10)
    else:
        tf.summary.image('input_0-3', batch_images[:,:,:,0:3], max_outputs=10)

    # FIRST LAYER kernels and all the trainable ops
    for op in tf.get_default_graph().get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        try:
            if 'kernel' in op.name and op.shape[2] == args.dimensions: # FIRST LAYER kernels
            	if args.dimensions ==1:
            			weights_transposed = tf.transpose(op, [3, 0, 1, 2])
	                    tf.summary.image(op.name + '_' + str(index), weights_transposed[:,:,:, 0], max_outputs=10)
            	else:
	                for index in xrange(int(op.shape[2]/3)):
	                    init=index*3
	                    final=(index+1)*3
	                    '''
	                     x_min = tf.reduce_min(weights)
	                      x_max = tf.reduce_max(weights)
	                      weights_0_to_1 = (weights - x_min) / (x_max - x_min)
	                      weights_0_to_255_uint8 = tf.image.convert_image_dtype (weights_0_to_1, dtype=tf.uint8)
	                    '''
	                    weights_transposed = tf.transpose(op, [3, 0, 1, 2])
	                    tf.summary.image(op.name + '_' + str(index), weights_transposed[:,:,:, init:final], max_outputs=10)
        except:
            pass
        # all the trainable ops
        tf.summary.histogram(op.name, op)




augmenter_seq = get_augmenter(name = 'DAVIS')


saver = tf.train.Saver(tf.global_variables())

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
            batch_size = batch_size * 2

        total_batch = int(training_samples / batch_size)

        # steps in every epoch
        for step in range(total_batch):
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            if augmentation :
                batch_x, batch_y = loader.get_batch(size=batch_size, train=True, percentage_noise=random.random()/12)
                batch_x = augmenter_seq.augment_images(batch_x)
            else:
                batch_x, batch_y = loader.get_batch(size=batch_size, train=True)

            batch_x = batch_x.astype(np.float16)/255 - 0.5


            train_feed_dict = {
                x: batch_x,
                label: batch_y,
                learning_rate: epoch_learning_rate,
                training_flag: True
            }

            _, loss = sess.run([train, cost], feed_dict=train_feed_dict)

            if step != 0 and step % 100 == 0:
                global_step += 100
                train_summary, train_accuracy = sess.run([merged, accuracy], feed_dict=train_feed_dict)
                # accuracy.eval(feed_dict=feed_dict)
                print("Step:", step, "Loss:", loss, "Training accuracy:", train_accuracy)
                writer_train.add_summary(train_summary, global_step=epoch)

                batch_x_test, batch_y_test = loader.get_batch(size=batch_size, train=False)
                batch_x_test = batch_x_test.astype(np.float16)/255 - 0.5

                test_feed_dict = {
                    x: batch_x_test,
                    label: batch_y_test,
                    learning_rate: epoch_learning_rate,
                    training_flag: False
                }

                test_summary, accuracy_rates = sess.run([merged, accuracy], feed_dict=test_feed_dict)
                writer_test.add_summary(test_summary, global_step=epoch)
                print('Step:', '%04d' % (step), '/ Accuracy =', accuracy_rates)

        print('Epoch:', '%04d' % (epoch + 1), '/ Accuracy =', accuracy_rates)
        saver.save(sess=sess, save_path='./model/dense.ckpt')
    # writer.add_summary(test_summary, global_step=epoch)

    saver.save(sess=sess, save_path='./model/dense.ckpt')