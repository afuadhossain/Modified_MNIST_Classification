# A Convolutional Neural Net 
#
# Parker King-Fournier
# 260556983

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import pickle
import scipy.ndimage as img
import io

tf.logging.set_verbosity(tf.logging.INFO)

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

#______________________A leftover binary normalization method______________________
def binarynormalization(arr, threshold):
    arr = arr.astype('float')

    for i in range(arr.shape[0]):
        minval = arr[i,...].min()
        maxval = arr[i,...].max()
        if minval!= maxval:
            arr[i,...] -= minval
            arr[i,...] /= (maxval-minval)



        for x in range(arr.shape[1]):
            for y in range(arr.shape[2]):
                if arr[i,x,y] >= threshold:
                    arr[i,x,y] = 1.0
                else:
                    arr[i,x,y] = 0.0
    return arr

#_________________________A Convolutional Neural Network_________________________
def cnn_model_fn(features, labels, mode):

    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=40,
      kernel_size=[9, 9],
      padding="same",
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
      bias_initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.05),
      activation=tf.nn.relu)
    # Max Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)
    # Normalization Layer #1
    normal1 = tf.layers.batch_normalization(
       inputs = pool1)


    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
      inputs=normal1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
      bias_initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.05),
      activation=tf.nn.relu)
    # Max Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)
    # Normalization Layer #2
    normal2 = tf.layers.batch_normalization(
       inputs = pool2)

    # Convolution Layer #3
    conv3 = tf.layers.conv2d(
      inputs=normal2,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
      bias_initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.05),
      activation=tf.nn.relu)
    # Max Pooling Layer #2
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[3, 3], strides=2)
    # Normalization Later #3
    normal3 = tf.layers.batch_normalization(
       inputs = pool3)

    # Convolution Later #4
    conv4 = tf.layers.conv2d(
      inputs=normal3,
      filters=96,
      kernel_size=[3, 3],
      padding="same",
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
      bias_initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.05),
      activation=tf.nn.relu)
    # Max Pooling Layer #4
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[3, 3], strides=2)


    # Flatten tensor into a batch of vectors
    pool4_flat = tf.reshape(pool4, [-1, 3 * 3 * 96])

    # Dense Layer #1
    dense1 = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
    # Add dropout operation; 0.5 probability that element will be kept
    dropout = tf.layers.dropout(
      inputs=dense1, rate=0.5, training=mode==tf.estimator.ModeKeys.TRAIN)

    # Dense Layer #2
    dense2 = tf.layers.dense(inputs=dropout, units=512, activation=tf.nn.relu)
    # Add dropout operation; 0.5 probability that element will be kept
    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.5, training=mode==tf.estimator.ModeKeys.TRAIN)

    #  Output Layer
    logits = tf.layers.dense(inputs=dropout2, units=82)


    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=82)
    loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)


    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


#_________________________________________MAIN_________________________________________
def main(unused_argv):

    # Read in pkl files
    x_in = open('../data/pkl/x_train_data.pkl','rb')
    y_in = open('../data/pkl/y_train_data.pkl', 'rb')
    print('loading pickled data...')
    x_train = pickle.load(x_in) # load from text
    y_train = pickle.load(y_in)
    print('done loading data!')
    y_train = np.asarray(y_train, dtype=np.int32)
    x_in.close()
    y_in.close()

    # Preprocess data
    x_train= binarynormalization(x_train,0.71)
    x_train = np.asarray(x_train, dtype=np.float32)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/comp551_convnet_model212")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

  # Create the Estimator
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_train},
        y=y_train,
        batch_size=64,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=500,
        hooks=[logging_hook])

    # Load training and eval data
    x_test_in = open('../data/pkl/x_test_data.pkl','rb')
    print('loading pickled data...')
    x_test = pickle.load(x_test_in) # load from text
    print('done loading data!')
    x_test = np.asarray(x_test, dtype=np.float32)
    x_test_in.close()

    # Make Predictions
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
          x={"x":x_test},
          num_epochs=1,
          shuffle=False)
    predictions = list(mnist_classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [p["classes"] for p in predictions]

    # Print them to a file
    output = io.open('first_pred.csv', 'w', encoding='utf-8')
    count = 1
    output.write(u'Id,Label\n')
    for x in predicted_classes:
        output.write(str(count) + u',' + str(x) + u'\n')
        count += 1
    output.close()

if __name__ == "__main__":
  tf.app.run()