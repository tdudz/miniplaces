import datetime
import argparse
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
from keras_models import alexnet
from keras_models import alexnet_bn
from keras_models import VGG16
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
num_iter = 2000

# Training Parameters
learning_rate = 0.00001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 100000
step_display = 50
step_save = 1000
gpus = args.gpus

# Construct dataloader
opt_data_train = {
    'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data/images/',
    'data_list': '../../data/train.txt',
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }

opt_data_val = {
    'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',
    'data_list': '../../data/val.txt',
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

print "Loading data loaders"
# loader_train = DataLoaderDisk(**opt_data_train)
# loader_val = DataLoaderDisk(**opt_data_val)
loader_train = DataLoaderH5(**opt_data_train)
loader_val = DataLoaderH5(**opt_data_val)

# Loss and accuracy functions
def sparse_categorical_crossentropy_with_logits(y_true, y_pred):
    return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

def top_5_acc(y_true, y_pred, k=5):
    # sparse_top_5_categorical_accuracy
    return K.mean(K.in_top_k(y_pred, K.cast(K.max(y_true, axis=-1), 'int32'), k), axis=-1)

def top_1_acc(y_true, y_pred, k=1):
    # sparse_top_1_categorical_accuracy
    return K.mean(K.in_top_k(y_pred, K.cast(K.max(y_true, axis=-1), 'int32'), k), axis=-1)

print "Initializing model on CPU"
with tf.device('/cpu:0'):
    model = alexnet_bn_keras((fine_size, fine_size, c))
    # model = VGG16(input_shape=(fine_size, fine_size, c))

parallel_model = multi_gpu_model(model, gpus=gpus)

opt = Adam(lr=learning_rate)
parallel_model.compile(loss=sparse_categorical_crossentropy_with_logits, optimizer=opt, metrics=[top_5_acc, top_1_acc])

# checkpointer = ModelCheckpoint(filepath='/data/keras_saved/weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_weights_only=True)

opt = Adam(lr=learning_rate)
parallel_model.compile(loss=sparse_categorical_crossentropy_with_logits, optimizer=opt, metrics=[top_5_acc, top_1_acc])

for i in xrange(num_iter):
    print "Iter:", i
    images_batch, labels_batch = loader_train.next_batch(batch_size)
    images_batch_val, labels_batch_val = loader_val.next_batch(batch_size) 
    parallel_model.fit(images_batch, labels_batch, batch_size=batch_size, epochs=1, verbose=1, validation_data=(images_batch_val, labels_batch_val))
    fname = 'keras-' + str(i) + '.h5'
    model.save_weights(fname)

print("Optimization Finished!")

# Evaluate on the whole validation set
print('Evaluation on the whole validation set...')
loader_val.reset()
loss = model.evaluate(images_batch_val, labels_batch_val, batch_size=val_size)
print "FINAL VALIDATION LOSS, TOP5, TOP1: " + loss
