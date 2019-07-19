# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TensorFlow/Keras model definition."""


import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.estimator import model_to_estimator
import glob


def keras_estimator(model_dir, config, learning_rate):
    """Creates a CNN using Keras.

    This function creates a CNN using TensorFlow's Keras API. The Keras model is
    converted to a Tensorflow Estimator so that it can be consumed by
    SageMaker's sagemaker.tensorflow.TensorFlow API.

    Args:
      model_dir: (str) File path where training files will be written.
      config: (tf.estimator.RunConfig) Configuration options to save model.
      learning_rate: (float) Gradient Descent learning rate.

    Returns:
      A keras.Model
    """

    # Input layer name must match the feature dictionary feeding the network
    # defined in the input_fn() / _parse_fun()
    inputs = Input(shape=(28, 28, 1), name='image_input')
    x = Conv2D(
        filters=32,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)(inputs)
    x = MaxPool2D(pool_size=(3, 3), strides=2)(x)
    x = Conv2D(
        filters=64,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2)(x)
    x = Flatten()(x)
    x = Dense(128, activation=tf.nn.relu)(x)
    x = Dropout(rate=0.4)(x)
    output = Dense(10, activation=tf.nn.softmax)(x)
    model = Model(inputs, output)

    # Compile model with learning parameters.
    optimizer = Adam(lr=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    # Converts the Keras model to a TensorFlow Estimator
    estimator = model_to_estimator(
        keras_model=model, model_dir=model_dir, config=config)
    return estimator


def input_fn(tfrecords_dir, batch_size, mode):
    """Reads TFRecords, parses them, and returns a dataset.

    Args:
      tfrecords_dir: (str) Directory containing TFRecords.
      batch_size: (int) Batch size.
      mode: (tf.estimator.ModeKeys) Estimator mode (PREDICT, EVAL, TRAIN).

    Returns:
        tf.data.Dataset
    """
    try:
        tfrecords_path = tfrecords_dir + "/*.tfrecords"
        tfrecords_file_queue = glob.glob(tfrecords_path)
        print("TFRecords File Queue Constructed: {}"
              .format(tfrecords_file_queue))

    except TypeError:
        raise ValueError("tfrecords_dir should contain a valid path but "
                         "instead contained: {}".format(tfrecords_path))

    dataset = tf.data.TFRecordDataset(tfrecords_file_queue).map(_parse_fn)

    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
        dataset = dataset.batch(batch_size)
    return dataset


def _parse_fn(example):
    """Parses a single MNIST TFRecord.

    This function parses a single MNIST TFRecord and returns a label and
    feature set.  It is intended to be called by the tf.data.Dataset
    transformation chain defined in input_fun().

    Args:
      example: Single row from a TFRecord.

    Returns:
        {'image_input': image}: {'image_input':tf.tensor} A feature dictionary
        with a single key 'image_input' that maps to a height x width x depth
        input tensor.
        label: (tf.int32) A scalar label.
    """

    feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'depth': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image_raw': tf.io.FixedLenFeature([], tf.string, default_value="")}

    example = tf.io.parse_single_example(example, feature_description)
    image = tf.decode_raw(example['image_raw'], tf.uint8)
    image = tf.cast(image, tf.float32)
    image /= 255
    image = tf.reshape(image,
                       [example['height'],
                        example['width'],
                        example['depth']])
    image.set_shape([28, 28, 1])
    label = tf.cast(example['label'], tf.int32)
    return {'image_input': image}, label


def serving_input_fn():
    """Serving Input Function for tf.Estimator inference.

    Allows the tf.Estimator to be serialized and provides a tensor placeholder
    for inference.

    Returns:
      tf.estimator.export.ServingInputReceiver
    """
    feature_spec = {'image_input': tf.placeholder(shape=[None, 28, 28, 1],
                                                  dtype=tf.float32)}
    return tf.estimator.export.ServingInputReceiver(feature_spec, feature_spec)
