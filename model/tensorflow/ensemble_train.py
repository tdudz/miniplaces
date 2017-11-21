import os, datetime
import numpy as np
import tensorflow as tf
import argparse
import tf_models
from DataLoader import *
from PIL import Image

# Command Line Argument Parsing
parser = argparse.ArgumentParser(description='TensorFlow Model Trainer')
parser.add_argument('--restore', help='whether to restore model or not', action='store_true', default=False)
args = parser.parse_args()

# Dataset Parameters
batch_size = 256
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 5
step_display = 1
step_save = 1

ensemble_path_save = '../../data/saved/ensemble'
ensemble_start_from = ''
alexnet_start_from = '../../data/saved/alexnet-1' 
resnet_start_from = '../../data/saved/resnet-1'

restore_model = args.restore

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

# loader_train = DataLoaderH5(**opt_data_train)
# loader_val = DataLoaderH5(**opt_data_val)
loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)

# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# global step
#global_step = tf.Variable(0, name='global_step', trainable=False)

# Construct alexnet model
with tf.variable_scope("alexnet_piece"):
    alexnet_logits = tf_models.alexnet_bn(x, keep_dropout, train_phase)

resnet_size = 18
num_classes = 100
with tf.variable_scope("resnet_piece"):
    resnet = tf_models.imagenet_resnet_v2(resnet_size, num_classes)
    resnet_logits = resnet(x, False)

with tf.variable_scope("ensemble"):
    wc = tf.Variable(tf.ones([200,100])/20000)
    logits = tf.concat([resnet_logits, alexnet_logits], 1)
    logits = tf.matmul(logits, wc)
    #logits = tf.contrib.layers.fully_connected(inputs=logits,num_outputs=num_classes)






# Evaluate model
accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))

# define initialization



some_vars = tf.global_variables()

other_vars = tf.trainable_variables()




alexnet_vars = {k.name[14:][:-2]:k for k in some_vars if k.name.startswith("alexnet_piece")}
resnet_vars = {k.name[13:][:-2]:k for k in some_vars if k.name.startswith("resnet_piece")}
ensemble_vars = {k.name[:][:-2]:k for k in some_vars if k.name.startswith("ensemble")}

ensemble_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
ensemble_train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(ensemble_loss,var_list=ensemble_vars)

init = tf.global_variables_initializer()



# variables_to_restore = tf.get_collection(tf.variables.VARIABLES_TO_RESTORE)#tf.contrib.slim.get_variables_to_restore(["ensemble","resnet_piece","alexnet_piece"])
# alexnet_varlist = {v.name[13:][:-2]: v for v in variables_to_restore if v.name[:12]=='alexnet_piece'}
# resnet_varlist = {v.name[12:][:-2]: v for v in variables_to_restore if v.name[:11]=='resnet_piece'}
# ensemble_varlist = {v.name[12:][:-2]: v for v in variables_to_restore if v.name[:11]=='ensemble'}

# define saver

alexnet_saver = tf.train.Saver(var_list=alexnet_vars)
resnet_saver = tf.train.Saver(var_list=resnet_vars)
ensemble_saver = tf.train.Saver(var_list=ensemble_vars)


# define summary writer
writer = tf.summary.FileWriter('./logs', graph=tf.get_default_graph())

# Launch the graph
with tf.Session() as sess:
    # Initialization
    alexnet_saver.restore(sess, alexnet_start_from)
    resnet_saver.restore(sess,resnet_start_from)

    if len(ensemble_start_from) > 0:
        ensemble_saver.restore(sess,ensemble_start_from)
    else:
        sess.run(init)
        step = 0
        print "Initialized new model"


    while step < training_iters:
        # Load a batch of training data
        images_batch, labels_batch = loader_train.next_batch(batch_size)

        
        if step % step_display == 0:
            print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            # Calculate batch loss and accuracy on training set
            l, acc1, acc5 = sess.run([ensemble_loss, accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False}) 
            print("-Iter " + str(step) + ", Training Loss= " + \
                  "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                  "{:.4f}".format(acc1) + ", Top5 = " + \
                  "{:.4f}".format(acc5))

            # Calculate batch loss and accuracy on validation set
            images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)    
            l, acc1, acc5 = sess.run([ensemble_loss, accuracy1, accuracy5], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1., train_phase: False}) 
            print("-Iter " + str(step) + ", Validation Loss= " + \
                  "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                  "{:.4f}".format(acc1) + ", Top5 = " + \
                  "{:.4f}".format(acc5))
        
        # Run optimization op (backprop)
        sess.run(ensemble_train_optimizer, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout, train_phase: True})
        
        step += 1
        
        # Save model
        if step % step_save == 0:
            ensemble_saver.save(sess, ensemble_path_save, global_step=step)
            print("Model saved at Iter %d !" %(step))
        
    print("Optimization Finished!")

    # Evaluate on the whole validation set
    print('Evaluation on the whole validation set...')
    num_batch = loader_val.size()//batch_size
    acc1_total = 0.
    acc5_total = 0.
    loader_val.reset()
    for i in range(num_batch):
        images_batch, labels_batch = loader_val.next_batch(batch_size)    
        acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
        acc1_total += acc1
        acc5_total += acc5
        print("Validation Accuracy Top1 = " + \
              "{:.4f}".format(acc1) + ", Top5 = " + \
              "{:.4f}".format(acc5))

    acc1_total /= num_batch
    acc5_total /= num_batch
    print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))
