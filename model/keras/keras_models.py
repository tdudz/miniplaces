import tensorflow as tf
from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Activation, Input, BatchNormalization, Conv2D, MaxPooling2D
from keras import regularizers
import numpy as np

def alexnet(input_shape, weights_path=None, keep_dropout=0.5):
    model = Sequential()
    model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(256, (5, 5), strides=(1, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(384, (3, 3), strides=(1, 1), activation='relu'))

    model.add(Conv2D(384, (3, 3), strides=(1, 1), activation='relu'))

    model.add(Conv2D(256, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=4096, activation='relu'))
    model.add(Dropout(keep_dropout))

    model.add(Dense(units=4096, activation='relu'))
    model.add(Dropout(keep_dropout))

    model.add(Dense(units=100))

    return model

def alexnet_bn(input_shape, weights_path=None, keep_dropout=0.5):
    model = Sequential()

    model.add(Conv2D(96,(11,11),strides=(4,4),padding='same',input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.9,center=True,scale=True))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(256, (5, 5), strides=(1, 1)))
    model.add(BatchNormalization(momentum=0.9,center=True,scale=True))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(384, (3, 3), strides=(1, 1)))
    model.add(BatchNormalization(momentum=0.9,center=True,scale=True))
    model.add(Activation('relu'))

    model.add(Conv2D(384, (3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, (3, 3), strides=(1, 1)))
    model.add(BatchNormalization(momentum=0.9,center=True,scale=True))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=4096))
    model.add(BatchNormalization(momentum=0.9,center=True,scale=True))
    model.add(Activation('relu'))
    model.add(Dropout(keep_dropout))

    model.add(Dense(units=4096))
    model.add(BatchNormalization(momentum=0.9,center=True,scale=True))
    model.add(Activation('relu'))
    model.add(Dropout(keep_dropout))

    model.add(Dense(units=100))

    return model

def VGG16(input_shape, weights_path=None, keep_dropout=0.5):

    inputs = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(inputs)
    x = BatchNormalization(momentum=0.9,center=True,scale=True)(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    x = BatchNormalization(momentum=0.9,center=True,scale=True)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    x = BatchNormalization(momentum=0.9,center=True,scale=True)(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
    x = BatchNormalization(momentum=0.9,center=True,scale=True)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
    x = BatchNormalization(momentum=0.9,center=True,scale=True)(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
    x = BatchNormalization(momentum=0.9,center=True,scale=True)(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
    x = BatchNormalization(momentum=0.9,center=True,scale=True)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
    x = BatchNormalization(momentum=0.9,center=True,scale=True)(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
    x = BatchNormalization(momentum=0.9,center=True,scale=True)(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
    x = BatchNormalization(momentum=0.9,center=True,scale=True)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
    x = BatchNormalization(momentum=0.9,center=True,scale=True)(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
    x = BatchNormalization(momentum=0.9,center=True,scale=True)(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
    x = BatchNormalization(momentum=0.9,center=True,scale=True)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, name='fc1')(x)
    x = BatchNormalization(momentum=0.9,center=True,scale=True)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(4096, name='fc2')(x)
    x = BatchNormalization(momentum=0.9,center=True,scale=True)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(units=100, name='predictions')(x)

    # Create model
    model = Model(inputs, x, name='vgg16')

    return model