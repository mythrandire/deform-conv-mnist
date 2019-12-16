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
from networks import get_cnn, get_df_cnn
from fetch_mnist import *
import matplotlib.pyplot as plt

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


# Define datapaths for models, outputs etc.
cnn_path = './models/cnn.h5'
plot_path = './figures/cnn_train_acc_val.png'

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
inputs, outputs = get_cnn()
model = Model(inputs=inputs, outputs=outputs)

# print model summary
model.summary()
optim = Adam(1e-3)
# optim = SGD(1e-3, momentum=0.99, nesterov=True)
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
plt.title('CNN Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(plot_path)
plt.show()

# Save the model weights
model.save_weights(cnn_path)

# Evaluate the performance of the model
model.load_weights(cnn_path, by_name=True)

val_loss, val_acc = model.evaluate_generator(
    test_gen, steps=validation_steps
)
print('Test accuracy for plain: ', val_acc)

val_loss_s, val_acc_s = model.evaluate_generator(
    test_scaled_gen, steps=validation_steps
)

print('Test accuracy for scaled: ', val_acc_s)
