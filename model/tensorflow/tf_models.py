import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
import numpy as np

def alexnet(x, keep_dropout):
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
        'bc1': tf.Variable(tf.zeros(96)),
        'bc2': tf.Variable(tf.zeros(256)),
        'bc3': tf.Variable(tf.zeros(384)),
        'bc4': tf.Variable(tf.zeros(256)),
        'bc5': tf.Variable(tf.zeros(256)),

        'bf6': tf.Variable(tf.zeros(4096)),
        'bf7': tf.Variable(tf.zeros(4096)),
        'bo': tf.Variable(tf.zeros(100))
    }

    # Conv + ReLU + LRN + Pool, 224->55->27
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 4, 4, 1], padding='SAME')
    conv1 = tf.nn.relu(tf.nn.bias_add(conv1, biases['bc1']))
    lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU + LRN + Pool, 27-> 13
    conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(tf.nn.bias_add(conv2, biases['bc2']))
    lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU, 13-> 13
    conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.relu(tf.nn.bias_add(conv3, biases['bc3']))

    # Conv + ReLU, 13-> 13
    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = tf.nn.relu(tf.nn.bias_add(conv4, biases['bc4']))

    # Conv + ReLU + Pool, 13->6
    conv5 = tf.nn.conv2d(conv4, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = tf.nn.relu(tf.nn.bias_add(conv5, biases['bc5']))
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # FC + ReLU + Dropout
    fc6 = tf.reshape(pool5, [-1, weights['wf6'].get_shape().as_list()[0]])
    fc6 = tf.add(tf.matmul(fc6, weights['wf6']), biases['bf6'])
    fc6 = tf.nn.relu(fc6)
    fc6 = tf.nn.dropout(fc6, keep_dropout)
    
    # FC + ReLU + Dropout
    fc7 = tf.add(tf.matmul(fc6, weights['wf7']), biases['bf7'])
    fc7 = tf.nn.relu(fc7)
    fc7 = tf.nn.dropout(fc7, keep_dropout)

    # Output FC
    out = tf.add(tf.matmul(fc7, weights['wo']), biases['bo'])
    
    return out

def alexnet_bn(x, keep_dropout, train_phase):
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

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
                      updates_collections=None,
                      is_training=train_phase,
                      reuse=None,
                      trainable=True,
                      scope=scope_bn)

########################################################
################### RESNET CODE ########################
########################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def batch_norm_relu(inputs, is_training, data_format):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=is_training, fused=True)
  inputs = tf.nn.relu(inputs)
  return inputs

def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)

def building_block(inputs, filters, is_training, projection_shortcut, strides,
                   data_format):
  """Standard building block for residual networks with BN before convolutions.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training, data_format)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)

  return inputs + shortcut

def bottleneck_block(inputs, filters, is_training, projection_shortcut,
                     strides, data_format):
  """Bottleneck block variant for residual networks with BN before convolutions.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first two convolutions. Note that the
      third and final convolution will use 4 times as many filters.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training, data_format)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)

  return inputs + shortcut

def block_layer(inputs, filters, block_fn, blocks, strides, is_training, name,
                data_format):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    is_training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block layer.
  """
  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = 4 * filters if block_fn is bottleneck_block else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides,
                    data_format)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, is_training, None, 1, data_format)

  return tf.identity(inputs, name)

def cifar10_resnet_v2_generator(resnet_size, num_classes, data_format=None):
  """Generator for CIFAR-10 ResNet v2 models.

  Args:
    resnet_size: A single integer for the size of the ResNet model.
    num_classes: The number of possible classes for image classification.
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.

  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the ResNet model.

  Raises:
    ValueError: If `resnet_size` is invalid.
  """
  if resnet_size % 6 != 2:
    raise ValueError('resnet_size must be 6n + 2:', resnet_size)

  num_blocks = (resnet_size - 2) // 6

  if data_format is None:
    data_format = (
        'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

  def model(inputs, is_training):
    """Constructs the ResNet model given the inputs."""
    if data_format == 'channels_first':
      # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
      # This provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=16, kernel_size=3, strides=1,
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')

    inputs = block_layer(
        inputs=inputs, filters=16, block_fn=building_block, blocks=num_blocks,
        strides=1, is_training=is_training, name='block_layer1',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=32, block_fn=building_block, blocks=num_blocks,
        strides=2, is_training=is_training, name='block_layer2',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=64, block_fn=building_block, blocks=num_blocks,
        strides=2, is_training=is_training, name='block_layer3',
        data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=8, strides=1, padding='VALID',
        data_format=data_format)
    inputs = tf.identity(inputs, 'final_avg_pool')
    inputs = tf.reshape(inputs, [-1, 64])
    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    inputs = tf.identity(inputs, 'final_dense')
    return inputs

  return model

def imagenet_resnet_v2_generator(block_fn, layers, num_classes,
                                 data_format=None):
  """Generator for ImageNet ResNet v2 models.

  Args:
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    layers: A length-4 array denoting the number of blocks to include in each
      layer. Each layer consists of blocks that take inputs of the same size.
    num_classes: The number of possible classes for image classification.
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.

  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the ResNet model.
  """
  if data_format is None:
    data_format = (
        'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

  def model(inputs, is_training):
    """Constructs the ResNet model given the inputs."""
    if data_format == 'channels_first':
      # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
      # This provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=64, kernel_size=7, strides=2,
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=3, strides=2, padding='SAME',
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')

    inputs = block_layer(
        inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
        strides=1, is_training=is_training, name='block_layer1',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
        strides=2, is_training=is_training, name='block_layer2',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
        strides=2, is_training=is_training, name='block_layer3',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
        strides=2, is_training=is_training, name='block_layer4',
        data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=7, strides=1, padding='VALID',
        data_format=data_format)
    inputs = tf.identity(inputs, 'final_avg_pool')
    inputs = tf.reshape(inputs,
                        [-1, 512 if block_fn is building_block else 2048])
    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    inputs = tf.identity(inputs, 'final_dense')
    return inputs

  return model

def imagenet_resnet_v2(resnet_size, num_classes, data_format=None):
  """Returns the ResNet model for a given size and number of output classes."""
  model_params = {
      18: {'block': building_block, 'layers': [2, 2, 2, 2]},
      34: {'block': building_block, 'layers': [3, 4, 6, 3]},
      50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
      101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
      152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
      200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
  }

  if resnet_size not in model_params:
    raise ValueError('Not a valid resnet_size:', resnet_size)

  params = model_params[resnet_size]
  return imagenet_resnet_v2_generator(
      params['block'], params['layers'], num_classes, data_format)

########################################################
########################################################
########################################################