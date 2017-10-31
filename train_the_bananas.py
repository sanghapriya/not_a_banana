import tensorflow as tf
from tensorflow.python.platform import gfile
import os
import numpy as np
import argparse
import sys
from sklearn import preprocessing
from read_image import prepare_data,read_image_array,read_single_image


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):

    x = tf.placeholder(tf.float32, shape=[None, 2352])
    y_ = tf.placeholder(tf.float32, shape=[None, 2])

    # First Convolution and Pooling Layer

    conv_weight_1 = weight_variable([5, 5, 3, 31])
    conv_bias_1 = bias_variable([31])

    x_image = tf.reshape(x, [-1, 28, 28, 3])
    conv_1_1 = conv2d(x_image, conv_weight_1)
    conv_1 = tf.nn.relu(conv2d(x_image, conv_weight_1) + conv_bias_1)
    pool_1 = max_pool_2x2(conv_1)

    # Second Convolution and Pooling layer

    conv_weight_2 = weight_variable([5, 5, 31, 64])
    conv_bias_2 = bias_variable([64])

    conv_2 = tf.nn.relu(conv2d(pool_1, conv_weight_2) + conv_bias_2)
    pool_2 = max_pool_2x2(conv_2)

    # First fully connected layer

    fc_weight_1 = weight_variable([7 * 7 * 64, 1024])
    fc_bias_1 = bias_variable([1024])

    pool_2_flat = tf.reshape(pool_2, [-1, 7*7*64])
    fc_1 = tf.nn.relu(tf.matmul(pool_2_flat, fc_weight_1) + fc_bias_1)

      # A drop out layer
    keep_prob = tf.placeholder(tf.float32)
    custom_fc1_drop = tf.nn.dropout(fc_1, keep_prob)

      # Second custom fully connected layer
    fc_weights_2 = weight_variable([1024,2])
    fc_bias_2 = bias_variable([2])
    fc_2 = tf.matmul(fc_1, fc_weights_2) + fc_bias_2

    y_conv = fc_2

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:

      sess.run(tf.global_variables_initializer())
      file_list, y_image_label = prepare_data(FLAGS.image_dir)

      le = preprocessing.LabelEncoder()
      y_one_hot = tf.one_hot(le.fit_transform(y_image_label),depth=2)

      x_feed = sess.run(read_image_array(file_list))
      y_feed = sess.run(y_one_hot)

      for i in range(200):
        if i % 10 == 0:
          train_accuracy = accuracy.eval(feed_dict={
              x:x_feed , y_: y_feed, keep_prob: 1.0})
          print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x:x_feed , y_: y_feed, keep_prob: 0.8})
      predicted = tf.argmax(y_conv, 1)

      if FLAGS.predict_image <> "":
                x_single_img = sess.run(read_single_image(FLAGS.predict_image))
                print("You got ",le.inverse_transform(sess.run(predicted,feed_dict={x:x_single_img})))



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
        '--image_dir',
        type=str,
        default='images',
        help='Path to folders of labeled images.'
  )
  parser.add_argument(
        '--predict_image',
        type=str,
        default="",
        help='Unknown image'
    )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
