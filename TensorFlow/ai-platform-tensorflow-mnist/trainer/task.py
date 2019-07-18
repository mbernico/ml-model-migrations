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
"""Trainer for MNIST CNN."""


import os
import argparse
import tensorflow as tf
import logging
import model


def get_args():
    """Command Line Argument parser.

        Returns:
            Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--steps',
        type=int,
        default=10,
        help='The number of steps to train for.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='The batch size to use during training.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.1,
        help='The learning rate that the optimizer will use.')
    # Input data and model directories.
    parser.add_argument(
        '--model_dir',
        type=str,
        default=os.environ.get('SM_MODEL_DIR'),
        help="Storage location for the estimator.")
    parser.add_argument(
        '--train',
        type=str,
        default=os.environ.get('SM_CHANNEL_TRAIN'),
        help='The location of the training data.')
    parser.add_argument(
        '--test',
        type=str,
        default=os.environ.get('SM_CHANNEL_EVAL'),
        help='The location of the testing data.')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    args = parser.parse_args()
    return args


def train_and_evaluate(args):
    """Trains, evaluates, and serializes the MNIST model defined in model.py

    Args:
      args: (Parsed arguments obj) An object containing all parsed arguments.
    """
    # Define running config.
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=6000)

    # Create estimator.
    estimator = model.keras_estimator(
        model_dir=args.model_dir,
        config=run_config,
        learning_rate=args.learning_rate)

    # Create TrainSpec.
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: model.input_fn(
            args.train,
            batch_size=args.batch_size,
            mode=tf.estimator.ModeKeys.TRAIN),
        max_steps=args.steps)

    # Create EvalSpec.
    if os.path.exists('/opt/ml/model'):  # exists if running in AWS SM Container
        container_model_output_dir = '/opt/ml/model'
    else:
        container_model_output_dir = 'exporter'

    exporter = tf.estimator.LatestExporter(container_model_output_dir,
                                           model.serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: model.input_fn(
            args.test,
            batch_size=args.batch_size,
            mode=tf.estimator.ModeKeys.EVAL),
        steps=600,
        exporters=exporter,
        start_delay_secs=10,
        throttle_secs=60)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    """Training task entry point.
    """
    args = get_args()
    print(args.model_dir)
    logging.getLogger("tensorflow").setLevel(args.verbosity)
    train_and_evaluate(args)
