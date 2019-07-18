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

