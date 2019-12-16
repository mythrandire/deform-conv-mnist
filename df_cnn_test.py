"""
# df_cnn_test.py
# test CNN with deformable convolution
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
df_path = './models/df_cnn.h5'
plot_path = './figures/df_cnn_train_acc_val.png'

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

# Deformable CNN
inputs, outputs = get_df_cnn(trainable=False)
model = Model(inputs=inputs, outputs=outputs)

# Use weights from regular CNN as initialization to save time
model.load_weights(cnn_path, by_name=True)

# Print model summary of DFCNN
model.summary()
optim = Adam(5e-4)
# optim = SGD(1e-4, momentum=0.99, nesterov=True)
loss = categorical_crossentropy
model.compile(optim, loss, metrics=['accuracy'])

# retain history object returned by method fit_generator
history = model.fit_generator(
    train_scaled_gen, steps_per_epoch=steps_per_epoch,
    epochs=20, verbose=1,
    validation_data=test_scaled_gen, validation_steps=validation_steps
)
# Save model weights
model.save_weights(df_path)

# Display all keys available
print(history.history.keys())

# Visualize training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('DFCNN Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(plot_path)
plt.show()

model.load_weights(df_path)

val_loss, val_acc = model.evaluate_generator(
    test_scaled_gen, steps=validation_steps
)
print('Test accuracy of deformable convolution with scaled images', val_acc)
# 0.9255

val_loss, val_acc = model.evaluate_generator(
    test_gen, steps=validation_steps
)
print('Test accuracy of deformable convolution with regular images', val_acc)
# 0.9727

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
