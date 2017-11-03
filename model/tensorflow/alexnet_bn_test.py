import tensorflow as tf
import argparse
import numpy as np
import scipy.misc
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *

# command line argument parsing
parser = argparse.ArgumentParser(description='Alexnet')
# parser.add_argument('--restore', nargs=1, help='directory to saved weights')

args = parser.parse_args()
# restore_dir = args.restore[0]

data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
                      updates_collections=None,
                      is_training=train_phase,
                      reuse=None,
                      trainable=True,
                      scope=scope_bn)
    
def alexnet(x, keep_dropout, train_phase):
    weights = {
        'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96], stddev=np.sqrt(2./(11*11*3)))),
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256], stddev=np.sqrt(2./(5*5*96)))),
        'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=np.sqrt(2./(3*3*256)))),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=np.sqrt(2./(3*3*384)))),
        'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),

        'wf6': tf.Variable(tf.random_normal([7*7*256, 4096], stddev=np.sqrt(2./(7*7*256)))),
        'wf7': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2./4096))),
        'wo': tf.Variable(tf.random_normal([4096, 100], stddev=np.sqrt(2./4096)))
    }

    biases = {
        'bo': tf.Variable(tf.ones(100))
    }

    # Conv + ReLU + Pool, 224->55->27
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 4, 4, 1], padding='SAME')
    conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU  + Pool, 27-> 13
    conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = batch_norm_layer(conv2, train_phase, 'bn2')
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU, 13-> 13
    conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = batch_norm_layer(conv3, train_phase, 'bn3')
    conv3 = tf.nn.relu(conv3)

    # Conv + ReLU, 13-> 13
    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = batch_norm_layer(conv4, train_phase, 'bn4')
    conv4 = tf.nn.relu(conv4)

    # Conv + ReLU + Pool, 13->6
    conv5 = tf.nn.conv2d(conv4, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = batch_norm_layer(conv5, train_phase, 'bn5')
    conv5 = tf.nn.relu(conv5)
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # FC + ReLU + Dropout
    fc6 = tf.reshape(pool5, [-1, weights['wf6'].get_shape().as_list()[0]])
    fc6 = tf.matmul(fc6, weights['wf6'])
    fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
    fc6 = tf.nn.relu(fc6)
    fc6 = tf.nn.dropout(fc6, keep_dropout)
    
    # FC + ReLU + Dropout
    fc7 = tf.matmul(fc6, weights['wf7'])
    fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
    fc7 = tf.nn.relu(fc7)
    fc7 = tf.nn.dropout(fc7, keep_dropout)

    # Output FC
    out = tf.add(tf.matmul(fc7, weights['wo']), biases['bo'])
    
    return out

print "LOADING TEST DATA..."
############## LOADING TEST DATA ####################
image_dir = '../../data/images/'
list_im = []
print_paths = []
with open('../../data/test.txt', 'r') as f:
    for line in f:
        path = line.rstrip().split(' ')[0]
        print_paths.append(path)
        list_im.append(os.path.join(image_dir, path))
list_im = np.array(list_im, np.object)
num = list_im.shape[0]
print('# Images found:', num)

images_batch = np.zeros((num, 224, 224, 3))
for i in xrange(num):
    image = scipy.misc.imread(list_im[i])
    image = scipy.misc.imresize(image, (256, 256))
    image = image.astype(np.float32)/255.
    image = image - data_mean

    offset_h = (256-224)//2
    offset_w = (256-224)//2

    images_batch[i, ...] =  image[offset_h:offset_h+224, offset_w:offset_w+224, :]
#######################################################
print "FINISHED LOADING TEST DATA"

x = tf.placeholder(tf.float32, [1, 224, 224, 3])
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

logits = alexnet(x, keep_dropout, train_phase)
top_k = tf.nn.top_k(logits, k=5)

# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()

with tf.Session() as sess:
    # saver.restore(sess, tf.train.latest_checkpoint(restore_dir))
    sess.run(init)
    print "Loaded model"

    im = 0
    with open('./eval.txt', 'w') as f:
        # out = sess.run([top_k], feed_dict={x: images_batch, keep_dropout: 1., train_phase: False})
        for image, filename in zip(images_batch, print_paths):
            image = np.reshape(image, [1, 224, 224, 3])
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

