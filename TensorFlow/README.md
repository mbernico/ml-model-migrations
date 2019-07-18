# Moving Your TensorFlow Model from AWS SageMaker to Google Cloud AI Platform

This example shows you how to migrate a model created for AWS SageMaker onto 
Google's Cloud AI Platform with as few changes to your existing code as 
possible. 

## Assumptions
1. You're currently using [AWS 'Script Mode'](https://sagemaker.readthedocs.io/en/stable/using_tf.html#training-with-tensorflow)
1. You're using Python 3
1. You're using TensorFlow 1.12 or newer

## Tensorflow MNIST on SageMaker
Start by understaning this SageMaker example, which creates a Convolutional Neural Network using Tensorflow, Keras, and SageMaker for training and inference
on the famous MNIST dataset.

[Creating MNIST TFRecords on S3](sagemaker-tensorflow-mnist/mnist-tfrecord-creator.ipynb) - This notebook shows you how to create TFRecords from the MNIST dataset on S3    
[Local Training and Inference](sagemaker-tensorflow-mnist/train_mnist_local.ipynb) - This notebook shows you how to do training and inference with the SageMaker SDK on your local machine.  
[Cloud Training and Inference](sagemaker-tensorflow-mnist/train_mnist_remote.ipynb)  - This notebook shows you how to do training and inference with SageMaker in the AWS cloud.

Both examples presented here use a python script created for the training task, located in `training_job/`.  In that folder you will find two files called
`task.py` and `model.py`.  

### model.py
This file implements the core pieces of the TensorFlow model including the 
following methods:

**keras_estimator()** - Constructs a CNN using the TensorFlow Keras API and converts model to a TensorFlow estimator.  
**input_fn()** - Processes and deserializes the TFRecords used for model input.  
**serving_input_fn()** - Defines the input the model will recieve for inference.  

### task.py
This file is responsible for parsing command line arguments, and providing a driver loop for model training.

The training script can be ran locally, without Sagemaker, from the command line. For example, to run for 10 steps:

```bash
task.py --train /path/to/sagemaker-tensorflow-mnist/data/train/ --test /path/to/sagemaker-tensorflow-mnist/data/test/ --steps=10
```

## Migration to Cloud AI Platform
Now that you understand how training and inference work on SageMaker, we can show you how to move your SageMaker model easily to Google Cloud AI Platform.



### Moving your data from AWS S3 to GCP

1. Install and configure the [GCP SDK](https://cloud.google.com/sdk/install)
1. Create a bucket on GCP to store your model data
    ```bash
    gsutil mb gs://ml-model-migration
    ```
1. Edit the file ~/.boto and insert these lines.
    ```bash
    [s3]
    # Note that we specify region as part of the host, as mentioned in the AWS docs:
    # http://docs.aws.amazon.com/general/latest/gr/rande.html#s3_region
    host = s3.<YOUR_REGION>.amazonaws.com
    use-sigv4 = True
    ``` 
1. Find these lines in the file ~/.boto.
     ```bash
     #aws_access_key_id=
     #aws_secret_access_key=
    ```
    Uncomment each line and add your AWS Access key and Secret to the file
1. Use `gsutil` to copy your data from S3 to GCS
    ```bash
    gsutil rsync -r s3://sagemaker-us-east-2-708267171719/sagemaker/ml-model-migration/data/mnist gs://ml-model-migration/mnist_data
    ```
 
### Updating your training script to run on Cloud AI Platform
When you start a SageMaker training job, by default SageMaker will copy your training data
from the S3 locations you specify to a directory in the training container. Your script finds that directory (typically /opt/ml/train and /opt/ml/eval) by using
the environment variables SM_CHANNEL_TRAIN and SM_CHANNEL_EVAL. Your script must implement functionality to read that directory and use it for training or eval/test.

When you start a Cloud AI Platform training job, your script must implement the functionality necessary to make the data available to the model. That might mean
caching it in the container, however for the TensorFlow framework ` tf.data.TFRecordDataset()` is able to consume data directly from a GCS bucket. 

This means the only substantial change required for your training script is changing `input_fn()` to read directly from GCS:

```python
def input_fn(tfrecords_file, batch_size, mode):
    """Reads TFRecords, parses them, and returns a dataset.

    Args:
      tfrecords_file: (str) GCS file containing TFRecords.
      batch_size: (int) Batch size.
      mode: (tf.estimator.ModeKeys) Estimator mode (PREDICT, EVAL, TRAIN).

    Returns:
        tf.data.Dataset
    """

    dataset = tf.data.TFRecordDataset(tfrecords_file).map(_parse_fn)

    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
        dataset = dataset.batch(batch_size)
    return dataset
```

With that change, your script should work on Cloud AI Platform

### Running your training script locally

Your modified training script can be tested locally by using the gcloud ai-platform cli.

```bash
export TRAIN_DATA=gs://ml-model-migration/mnist_data/train/mnist_train.tfrecords
export EVAL_DATA=gs://ml-model-migration/mnist_data/test/mnist_test.tfrecords
export MODEL_DIR=output

gcloud ai-platform local train \
--module-name trainer.task \
--package-path trainer \
-- \
--train $TRAIN_DATA \
--test $EVAL_DATA 
--steps 1000 \ 
--learning_rate 0.001 \
--batch_size 100 \
--verbosity 'INFO' 
```

### Training your model on GCP Cloud AI Platform

```bash
export REGION=us-central1
export JOB_NAME=MNIST_EXAMPLE_$(date +%Y%m%d%H%M%S%s)
export OUTPUT_PATH=$BUCKET_NAME/$JOB_NAME

gcloud ai-platform jobs submit training $JOB_NAME \
--stream-logs \
--runtime-version 1.12 \
--module-name trainer.task \
--package-path trainer \
--python-version 3.5 \
--region $REGION \
--scale-tier BASIC_GPU \
--job-dir $OUTPUT_PATH \
-- \
--train $TRAIN_DATA \
--test $EVAL_DATA \
--steps 1000 \
--learning_rate 0.001 \
--batch_size 100 \
--verbosity 'INFO'
```

