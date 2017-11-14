import tensorflow as tf
import argparse
import numpy as np
import scipy.misc
import os
from keras import backend as K
from models import alexnet_bn_keras
from DataLoader import *

# command line argument parsing
parser = argparse.ArgumentParser(description='Alexnet Batch Normalization')
parser.add_argument('--model', nargs=1, help='directory to saved weights')
args = parser.parse_args()
weights = args.model[0]

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

# TODO: move to H5 loading
loader_test = DataLoaderTestDisk(**data_test)

# construct model
model = alexnet_bn_keras((fine_size, fine_size, c)) 

parallel_model = multi_gpu_model(model, gpus=2)

parallel_model.load_weights(weights)
print "Loaded weights from file"

images = loader_test.get_test_images()
paths = loader_test.get_file_list()

im = 0
with open('./eval.txt', 'w') as f:
    for image, filename in zip(images, paths):
        image = np.reshape(image, [1, fine_size, fine_size, c])
        logits = model.predict(image, batch_size=1, verbose=0)
        top_values, top_indices = K.get_session().run(tf.nn.top_k(logits, k=5))
        preds = ""
        for prediction in top_indices[0]:
            preds += str(prediction) + " "
        preds = preds[:-1]
        f.write(filename + " " + preds + "\n")
        im +=1
        if im % 200 == 0:
            print "TESTED", im, "IMAGES"

print "FINISHED EVALUATING TEST SET"