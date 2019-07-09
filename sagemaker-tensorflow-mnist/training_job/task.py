import os
import argparse
import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam
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
        help="Storage location for the estimator.")
    parser.add_argument(
        '--train',
        type=str,
        default=os.environ.get('SM_CHANNEL_TRAIN'),
        help='The location of the training data.')
    parser.add_argument(
        '--test',
        type=str,
        default=os.environ.get('SM_CHANNEL_TEST'),
        help='The location of the testing data.')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    return parser.parse_args()


def train_and_evaluate(hparams):
    """Trains, evaluates, and serializes the MNIST model defined in model.py

    Args:
      hparams: (tensorflow.contrib.training.python.training.hparam) A container
      class that holds parameters and hyperparameters for model training.
    """
    # Define running config.
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=500)

    # Create estimator.
    estimator = model.keras_estimator(
        model_dir=hparams.model_dir,
        config=run_config,
        learning_rate=hparams.learning_rate)

    # Create TrainSpec.
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: model.input_fn(
            hparams.train,
            batch_size=hparams.batch_size,
            mode=tf.estimator.ModeKeys.TRAIN),
        max_steps=hparams.steps)

    # Create EvalSpec.
    exporter = tf.estimator.LatestExporter('exporter', model.serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: model.input_fn(
            hparams.test,
            batch_size=hparams.batch_size,
            mode=tf.estimator.ModeKeys.EVAL),
        steps=None,
        exporters=exporter,
        start_delay_secs=10,
        throttle_secs=10)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    """Training task entry point.
    """
    args = get_args()
    logging.getLogger("tensorflow").setLevel(args.verbosity)
    train_and_evaluate(hparam.HParams(**args.__dict__))
