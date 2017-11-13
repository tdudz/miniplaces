import tensorflow as tf
import argparse
import numpy as np
import scipy.misc
import os
from keras import backend as K
from alexnet import alexnet_bn_keras
from DataLoader import *

# command line argument parsing
parser = argparse.ArgumentParser(description='Alexnet Batch Normalization')
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
    # 'data_h5': 'miniplaces_256_test.h5',
    'data_root': '../../data/images/',
    'data_list': '../../data/test.txt',
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

loader_test = DataLoaderTestDisk(**data_test)

# function for getting top k outputs using tf backend (takes in logits)
def top_k(logits, k=5):
    return K.top_k(logits, k=k)

# construct model
model = alexnet_bn_keras((fine_size, fine_size, c)) 

images = loader_test.get_test_images()
paths = loader_test.get_file_list()

im = 0
with open('./eval.txt', 'w') as f:
    for image, filename in zip(images, paths):
        image = np.reshape(image, [1, fine_size, fine_size, c])
        logits = model.predict(image, batch_size=1, verbose=1)
        out = top_k(logits)
        print out

    # sess.run(init)
    # print "Loaded model"

    # images = loader_test.get_test_images()
    # paths = loader_test.get_file_list()

    # # processess image by image
    # im = 0
    # with open('./eval.txt', 'w') as f:
    #     for image, filename in zip(images, paths):
    #         image = np.reshape(image, [1, fine_size, fine_size, c])
    #         out = sess.run([top_k], feed_dict={x: image, keep_dropout: 1., train_phase: False})
    #         import pdb; pdb.set_trace()
    #         print out
    #         preds = ""
    #         for prediction in out[0][1][0]:
    #             preds += str(prediction) + " "
    #         preds = preds[:-1]
    #         f.write(filename + " " + preds + "\n")
    #         im +=1
    #         if im % 250 == 0:
    #             print "TESTED", im, "IMAGES"

print "FINISHED EVALUATING TEST SET"

