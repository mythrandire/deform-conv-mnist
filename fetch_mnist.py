"""
Author: Dwiref Oza
2019
Fetch MNIST data
"""

from __future__ import absolute_import, division
import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator

def fetch_mnist():
    '''
    Load MNIST data and normalize it
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X_train = x_train.astype('float32') / 255
    X_test = x_test.astype('float32') / 255
    X_train = X_train[..., None]
    X_test = X_test[..., None]
    Y_train = keras.utils.to_categorical(y_train, 10)
    Y_test = keras.utils.to_categorical(y_test, 10)

    return (X_train, Y_train), (X_test, Y_test)


def get_gen(set_name, batch_size, translate, scale, shuffle=True):
    '''
    Image data generator class for batches with scaling and translation as
    augmentation
    '''
    if set_name == 'train':
        (X, Y), _ = fetch_mnist()
    elif set_name == 'test':
        _, (X, Y) = fetch_mnist()

    image_gen = ImageDataGenerator(
        zoom_range=scale,
        width_shift_range=translate,
        height_shift_range=translate
    )
    gen = image_gen.flow(X, Y, batch_size=batch_size, shuffle=shuffle)
    return gen
