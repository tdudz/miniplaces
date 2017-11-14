import tensorflow as tf
import argparse
import numpy as np
import scipy.misc
import os
from models import alexnet_bn, batch_norm_layer
from DataLoader import *

# command line argument parsing
parser = argparse.ArgumentParser(description='TensorFlow Model Evaluator')
# parser.add_argument('--model', nargs=1, help='directory to saved weights')
args = parser.parse_args()
# restore_dir = args.restore

# Dataset Parameters
batch_size = 256
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Construct dataloader
data_test = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data/images/',
    'data_list': '../../data/test.txt',
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

loader_test = DataLoaderTestDisk(**data_test)

# tf Graph input
x = tf.placeholder(tf.float32, [1, fine_size, fine_size, c])
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# construct model
logits = alexnet_bn(x, keep_dropout, train_phase) 
top_k = tf.nn.top_k(logits, k=5)

# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()

# launch the inference graph
with tf.Session() as sess:
    # saver.restore(sess, tf.train.latest_checkpoint(restore_dir))
    sess.run(init)
    print "Loaded model"

    images = loader_test.get_test_images()
    paths = loader_test.get_file_list()

    # TODO: make this 10x less ugly
    # processess image by image
    im = 0
    with open('./eval.txt', 'w') as f:
        for image, filename in zip(images, paths):
            image = np.reshape(image, [1, fine_size, fine_size, c])
            out = sess.run([top_k], feed_dict={x: image, keep_dropout: 1., train_phase: False})
            preds = ""
            for prediction in out[0][1][0]:
                preds += str(prediction) + " "
            preds = preds[:-1]
            f.write(filename + " " + preds + "\n")
            im +=1
            if im % 250 == 0:
                print "TESTED", im, "IMAGES"

print "FINISHED EVALUATING TEST SET"

