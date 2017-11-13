import datetime
import argparse
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from alexnet import alexnet_keras
from DataLoader import *

# command line argument parsing
parser = argparse.ArgumentParser(description='Alexnet Batch Normalization (Keras)')
parser.add_argument('--gpus', nargs='?', help='number of GPUs to train with', type=int, default=0)
args = parser.parse_args()

# Dataset Parameters
batch_size = 256
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Keras Parameters
train_size = 100000
val_size = 10000

# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 100000
step_display = 50
step_save = 1000
gpus = args.gpus

# Construct dataloader
opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data/images/',
    'data_list': '../../data/train.txt',
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }

opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',
    'data_list': '../../data/val.txt',
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)
#loader_train = DataLoaderH5(**opt_data_train)
#loader_val = DataLoaderH5(**opt_data_val)

def sparse_top_5_categorical_accuracy(y_true, y_pred, k=5):
    return K.mean(K.in_top_k(y_pred, K.cast(K.max(y_true, axis=-1), 'int32'), k), axis=-1)

def sparse_top_1_categorical_accuracy(y_true, y_pred, k=1):
    return K.mean(K.in_top_k(y_pred, K.cast(K.max(y_true, axis=-1), 'int32'), k), axis=-1)

with tf.device('/cpu:0'):
    model = alexnet_keras((fine_size, fine_size, c))

model = multi_gpu_model(model, gpus=gpus)

opt = Adam(lr=learning_rate)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=[sparse_top_5_categorical_accuracy, sparse_top_1_categorical_accuracy])


print "LOADING TEST AND VAL SET"
images_batch, labels_batch = loader_train.next_batch(train_size)
images_batch_val, labels_batch_val = loader_val.next_batch(val_size) 
print "FITTING MODEL"
model.fit(images_batch, labels_batch, batch_size=256, epochs=128, verbose=1, validation_data=(images_batch_val, labels_batch_val))

print("Optimization Finished!")

# Evaluate on the whole validation set
# print('Evaluation on the whole validation set...')
# num_batch = loader_val.size() // batch_size
# loader_val.reset()
# for i in range(num_batch):
#     images_batch, labels_batch = loader_val.next_batch(batch_size) 
#     loss = model.evaluate(images_batch, labels_batch, batch_size=batch_size)
