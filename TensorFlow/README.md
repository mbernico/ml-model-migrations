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
 
### Updating your training script to run on Cloud AI Platform.
There are very few changes you will need to make, in order for your AWS SageMaker Script to be runnable on Google Cloud AI Platform. The Key differences are
1. How the code is called.
2. How the trained model is exported and checkpointed
3. How data is fed to the model.

You can read more about each of these below.

### Running your code.
On Cloud AI Platform your code is expected to be bundled into a module. To bundle your code into a module you may need to:
1. Include an empty `__init__.py` in the same directory as your `model.py` and `task.py` files.
2. Adjust relative inputs in your module.

Once you've made these changes, you can test them by calling your code as a module (e.g. `python3 -m trainer.task` ) instead of as a script (e.g. `python3 trainer/task.py`).

### Adjusting how your model is exported and checkpointed.
On AWS SageMaker your model should be exported into `/opt/ml/model` in the container.  When the training job ends the serialized model will be copied out of the container
and onto S3 where the `sagemaker.tensorflow` object can find it and hand it off to a serving container for inference.

In addition to serializing your model in `/opt/ml/model` your script is also expected to implement a command line argument `model_dir` that is an S3 location that can
be used for model checkpoints.

When using Google Cloud AI Platform, your code is expected to implement a command line argument `job_dir` that represents a GCS bucket and path where both exported models
and checkpoint files can be stored.

1. Accept the command line parameter job_dir in your code.
1. Export models to the path specified in job_dir.
1. Model checkpoints should also be saved in job_dir.

#### Data
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
export BUCKET_NAME=gs://<YOUR-BUCKET-NAME>
export TRAIN_DATA=$BUCKET_NAME/mnist_data/train/mnist_train.tfrecords
export EVAL_DATA=$BUCKET_NAME/mnist_data/test/mnist_test.tfrecords
export OUTPUT_PATH=output

gcloud ai-platform local train \
--module-name trainer.task \
--package-path trainer \
-- \
--train $TRAIN_DATA \
--test $EVAL_DATA \
--steps 1000 \
--learning-rate 0.001 \
--batch-size 100 \
--verbosity 'INFO' \
--job-dir $OUTPUT_PATH
```

### Training your model on GCP Cloud AI Platform
With the modifications above, you're now ready to train on Google Cloud AI Platform.

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
--train-file $TRAIN_DATA \
--test-file $EVAL_DATA \
--steps 12000 \
--learning-rate 0.001 \
--batch-size 100 \
--verbosity 'INFO'
```

You can read more about using the [gcloud ai-platform CLI here](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction).

### Inference on GCP

1. Your model should have been exported to $OUTPUT_PATH. List the files in that bucket and find the model you want to serve. They are stored by timestamp, so the 
largest timestamp is the last checkpoint.

    Once you find the model you want, copy that path and set $MODEL_BINARIES to the model location.

    ```bash
    gsutil ls -r $OUTPUT_PATH/export
    export MODEL_BINARIES=gs://ml-model-migration/MNIST_EXAMPLE_201907191128471563553727/export/exporter/1563554211/
    ```
1. Define the model in Cloud AI Platform. 

    ```bash
    export MODEL_NAME=mnist
    export REGION=us-central1
    gcloud ai-platform models create $MODEL_NAME --regions=$REGION
    ```
    
1. Finally you will define the model version. Models can contain multiple versions, each version is related to a saved model binary.

    ```bash
    gcloud ai-platform versions create v1 \
    --model $MODEL_NAME \
    --origin $MODEL_BINARIES \
    --runtime-version 1.12
    ```
    
The model is now available as an endpoint on Cloud AI Platform.