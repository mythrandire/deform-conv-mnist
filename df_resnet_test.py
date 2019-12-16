"""
# cnn_test.py
# test normal CNN
#
# In your shell execute:
# export PYTHONPATH=~/Documents/DL/proj_test/deform-conv
# before running this code.
"""

from __future__ import division
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, SGD
from keras.utils import plot_model
from layers_temp import ConvOffset2D
from callbacks import TensorBoard
from networks import get_cnn, get_df_cnn, get_mini_df_resnet
from fetch_mnist import *
import matplotlib.pyplot as plt

from keras.layers.merge import add
from keras.activations import relu, softmax
from keras.models import Model
from keras import regularizers

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


# Define datapaths for models, outputs etc.
cnn_path = './models/df_resnet.h5'
plot_path = './figures/df_resnet_train_acc_val.png'

# Configure data for training and testing
batch_size = 32
n_train = 60000
n_test = 10000
steps_per_epoch = int(np.ceil(n_train / batch_size))
validation_steps = int(np.ceil(n_test / batch_size))

train_gen = get_gen(
    'train', batch_size=batch_size,
    scale=(1.0, 1.0), translate=0.0,
    shuffle=True
)
test_gen = get_gen(
    'test', batch_size=batch_size,
    scale=(1.0, 1.0), translate=0.0,
    shuffle=False
)
train_scaled_gen = get_gen(
    'train', batch_size=batch_size,
    scale=(1.0, 2.5), translate=0.2,
    shuffle=True
)
test_scaled_gen = get_gen(
    'test', batch_size=batch_size,
    scale=(1.0, 2.5), translate=0.2,
    shuffle=False
)


# Get and compile CNN model
inputs, outputs = get_mini_df_resnet(trainable = False)
model = Model(inputs=inputs, outputs=outputs)

# print model summary
model.summary()
# optim = Adam(1e-3)
optim = SGD(1e-3, momentum=0.9, nesterov=True)
loss = categorical_crossentropy
model.compile(optim, loss, metrics=['accuracy'])

# model.fit_generator returns a history object which lets you
# track the training history
# Model is trained using image data generator instance train_gen
# Validation using test_gen

history = model.fit_generator(
    train_gen, steps_per_epoch=steps_per_epoch,
    epochs=10, verbose=1,
    validation_data=test_gen, validation_steps=validation_steps
)

# Show the keys available in the history object
print(history.history.keys())

# Visualize training accuracy against validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Mini ResNet w/ deformable conv. Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(plot_path)
plt.show()

val_loss, val_acc = model.evaluate_generator(
    test_scaled_gen, steps=validation_steps
)
print('Test accuracy of deformable convolution with scaled images', val_acc)

val_loss, val_acc = model.evaluate_generator(
    test_gen, steps=validation_steps
)
print('Test accuracy of deformable convolution with regular images', val_acc)

deform_conv_layers = [l for l in model.layers if isinstance(l, ConvOffset2D)]

Xb, Yb = next(test_gen)
for l in deform_conv_layers:
    print(l)
    _model = Model(inputs=inputs, outputs=l.output)
    offsets = _model.predict(Xb)
    offsets = offsets.reshape(offsets.shape[0], offsets.shape[1], offsets.shape[2], -1, 2)
    print(offsets.min())
    print(offsets.mean())
    print(offsets.max())
