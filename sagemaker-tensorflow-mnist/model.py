import numpy as np

import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.estimator import model_to_estimator


def keras_estimator(model_dir, config, learning_rate):
    """Creates a Keras Sequential model with layers.
    Args:
      model_dir: (str) file path where training files will be written.
      config: (tf.estimator.RunConfig) Configuration options to save model.
      learning_rate: (int) Learning rate.
    Returns:
      A keras.Model
    """
    inputs = Input(shape=(28,28,1), name='image_input')
    x = Conv2D(
        filters=32,
        kernel_size=[5,5],
        padding='same',
        activation=tf.nn.relu)(inputs)
    x = MaxPool2D(pool_size=(2,2), strides=2)(x)
    x = Conv2D(
        filters=64,
        kernel_size=[5,5],
        padding='same',
        activation=tf.nn.relu)(x)
    x = MaxPool2D(pool_size=(2,2), strides=2)(x)
    x = Flatten()(x)
    x = Dense(1024, activation=tf.nn.relu)(x)
    x = Dropout(rate=0.4)(x)
    output = Dense(10, activation=tf.nn.softmax)(x)
    model = Model(inputs, output)

    # Compile model with learning parameters.
    optimizer = Adam(lr=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    estimator = model_to_estimator(
        keras_model=model, model_dir=model_dir, config=config)
    return estimator


def input_fn(tfrecords_path, batch_size, mode):
    dataset = tf.data.TFRecordDataset(tfrecords_path).map(_parse_fn)

    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
        dataset = dataset.batch(batch_size)
    return dataset


def _parse_fn(example):

    feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'depth': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image_raw': tf.io.FixedLenFeature([], tf.string, default_value="")}

    example = tf.io.parse_single_example(example, feature_description)
    image = tf.decode_raw(example['image_raw'], tf.uint8)
    image = tf.reshape(image,
                       [example['height'],
                        example['width'],
                        example['depth']])
    image.set_shape([28, 28, 1])
    label = tf.cast(example['label'], tf.int32)
    return {'image_input': image}, label


def serving_input_fn():
    feature_spec = {'image_input': tf.placeholder(shape=[None,28,28,1], dtype=tf.float32)}
    return tf.estimator.export.ServingInputReceiver(feature_spec,feature_spec)
