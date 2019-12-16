from __future__ import absolute_import, division


from keras.layers import Input, Conv2D, Activation, GlobalAvgPool2D, Dense, BatchNormalization, GlobalAveragePooling2D
from layers_temp import ConvOffset2D
from keras.layers.merge import add
from keras.activations import relu, softmax
from keras.models import Model
from keras import regularizers

def get_cnn():
    inputs = l = Input((28, 28, 1), name='input')

    # conv11
    l = Conv2D(32, (3, 3), padding='same', name='conv11')(l)
    l = Activation('relu', name='conv11_relu')(l)
    l = BatchNormalization(name='conv11_bn')(l)

    # conv12
    l = Conv2D(64, (3, 3), padding='same', strides=(2, 2), name='conv12')(l)
    l = Activation('relu', name='conv12_relu')(l)
    l = BatchNormalization(name='conv12_bn')(l)

    # conv21
    l = Conv2D(128, (3, 3), padding='same', name='conv21')(l)
    l = Activation('relu', name='conv21_relu')(l)
    l = BatchNormalization(name='conv21_bn')(l)

    # conv22
    l = Conv2D(128, (3, 3), padding='same', strides=(2, 2), name='conv22')(l)
    l = Activation('relu', name='conv22_relu')(l)
    l = BatchNormalization(name='conv22_bn')(l)

    # out
    l = GlobalAvgPool2D(name='avg_pool')(l)
    l = Dense(10, name='fc1')(l)
    outputs = l = Activation('softmax', name='out')(l)

    return inputs, outputs


def get_df_cnn(trainable):
    inputs = l = Input((28, 28, 1), name='input')

    # conv11
    l = Conv2D(32, (3, 3), padding='same', name='conv11', trainable=trainable)(l)
    l = Activation('relu', name='conv11_relu')(l)
    l = BatchNormalization(name='conv11_bn')(l)

    # conv12
    l_offset = ConvOffset2D(32, name='conv12_offset')(l)
    l = Conv2D(64, (3, 3), padding='same', strides=(2, 2), name='conv12', trainable=trainable)(l_offset)
    l = Activation('relu', name='conv12_relu')(l)
    l = BatchNormalization(name='conv12_bn')(l)

    # conv21
    l_offset = ConvOffset2D(64, name='conv21_offset')(l)
    l = Conv2D(128, (3, 3), padding='same', name='conv21', trainable=trainable)(l_offset)
    l = Activation('relu', name='conv21_relu')(l)
    l = BatchNormalization(name='conv21_bn')(l)

    # conv22
    l_offset = ConvOffset2D(128, name='conv22_offset')(l)
    l = Conv2D(128, (3, 3), padding='same', strides=(2, 2), name='conv22', trainable=trainable)(l_offset)
    l = Activation('relu', name='conv22_relu')(l)
    l = BatchNormalization(name='conv22_bn')(l)

    # out
    l = GlobalAvgPool2D(name='avg_pool')(l)
    l = Dense(10, name='fc1', trainable=trainable)(l)
    outputs = l = Activation('softmax', name='out')(l)

    return inputs, outputs

def resblock(n_output, upscale=False):

    def f(x):

        # first pre-actvation
        h = BatchNormalization()(x)
        h = Activation(relu)(h)
        # first convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same')(h)

        # second pre-activation
        h = BatchNormalization()(x)
        h = Activation(relu)(h)
        # second convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same')(h)

        if upscale:
            # 1x1 conv2d
            f = Conv2D(kernel_size=1, filters=n_output, strides=1, padding='same')(x)
        else:
            # identity
            f = x
        return add([f, h])
    return f

def get_mini_resnet():

    inputs = Input((28, 28, 1), name='input')
    # first conv2d with post-activation to transform the input data to some reasonable form
    x = Conv2D(kernel_size=3, filters=16, strides=1, padding='same', name = 'c_in')(inputs)
    x = BatchNormalization(name = 'bn_1')(x)
    x = Activation(relu)(x)
    # F_1
    x = resblock(16)(x)
    # F_2
    x = resblock(16)(x)
    # last activation of the entire network's output
    x = BatchNormalization()(x)
    x = Activation(relu)(x)
    x = GlobalAveragePooling2D()(x)

    # dropout for more robust learning
    # last softmax layer
    x = Dense(units=10)(x)
    outputs = x = Activation(softmax, name= 'output_layer')(x)

    return inputs, outputs

def df_resblock(n_output, trainable, upscale=False):

    def f(x):

        h = BatchNormalization()(x)
        h = Activation(relu)(h)
        h_offset = ConvOffset2D(n_output)(h)
        h = Conv2D(n_output, (3, 3), strides = 1, padding = 'same', trainable = trainable)(h_offset)

        # second pre-activation
        h = BatchNormalization()(x)
        h = Activation(relu)(h)
        h_offset = ConvOffset2D(n_output)(h)
        # second convolution
        h = Conv2D(n_output, (3, 3), strides=1, padding='same', trainable = trainable)(h_offset)

        if upscale:
            # 1x1 conv2d
            f = Conv2D(kernel_size=1, filters=n_output, strides=1, padding='same')(x)
        else:
            # identity
            f = x
        return add([f, h])
    return f


def get_mini_df_resnet(trainable):

    inputs = Input((28, 28, 1), name='input')
    # first conv2d with post-activation to transform the input data to some reasonable form
    x = Conv2D(kernel_size=3, filters=16, strides=1, padding='same', name = 'c_in')(inputs)
    x = BatchNormalization(name = 'bn_1')(x)
    x = Activation(relu)(x)
    # F_1
    x = df_resblock(16, trainable)(x)
    # F_2
    x = df_resblock(16, trainable)(x)
    # last activation of the entire network's output
    x = BatchNormalization()(x)
    x = Activation(relu)(x)
    x = GlobalAveragePooling2D()(x)

    # last softmax layer
    x = Dense(units=10)(x)
    outputs = x = Activation(softmax, name= 'output_layer')(x)

    return inputs, outputs
