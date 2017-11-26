

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os.path
import re
import sys
import tarfile
from read_image import prepare_data,read_image_array,read_single_image
import numpy as np
from six.moves import urllib
import tensorflow as tf
from sklearn import preprocessing

FLAGS = None

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def maybe_download_and_extract():
    """Download and extract model tar file."""
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
              sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                  filename, float(count * block_size) / float(total_size) * 100.0))
              sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def convert_images_to_bottlenecks(img_dir):

    maybe_download_and_extract()
    path = os.path.join(FLAGS.model_dir, 'classify_image_graph_def.pb')
    with tf.gfile.FastGFile(path, 'rb') as file:

        graph_def = tf.GraphDef()
        graph_def.ParseFromString(file.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        file_list,y_ = prepare_data(img_dir)
        bottleneck_list = []
        for file in file_list:
            transfer_layer = sess.graph.get_tensor_by_name("pool_3:0")
            print(" Creating bottleneck for : ",file)
            bottleneck = tf.reshape(np.squeeze(sess.run(transfer_layer,
            feed_dict={'DecodeJpeg/contents:0': tf.gfile.FastGFile(file, 'rb').read()})),[1,2048])
            bottleneck_list.append(bottleneck)

        bottleneck_list = tf.reshape(tf.stack(bottleneck_list),[len(file_list),2048])

        return bottleneck_list,y_


def convert_single_image_to_bottlenecks(image):


    path = os.path.join(FLAGS.model_dir, 'classify_image_graph_def.pb')
    with tf.gfile.FastGFile(path, 'rb') as file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(file.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        transfer_layer = sess.graph.get_tensor_by_name("pool_3:0")
        print(" Creating bottleneck for : ",image)
        bottleneck = tf.reshape(np.squeeze(sess.run(transfer_layer,
        feed_dict={'DecodeJpeg/contents:0': tf.gfile.FastGFile(image, 'rb').read()})),[1,2048])

        return bottleneck





def main(_):

     # First custom fully connected layer
      bottleneck_layer = tf.placeholder(tf.float32, shape=[None, 2048])
      y_ = tf.placeholder(tf.float32, shape=[None, 2])

      custom_fc_weights_1 = weight_variable([2048,1024])
      custom_fc_bias_1 = bias_variable([1024])
      custom_fc_layer_1 = tf.matmul(bottleneck_layer, custom_fc_weights_1) + custom_fc_bias_1

     # A drop out layer
      keep_prob = tf.placeholder(tf.float32)
      custom_fc1_drop = tf.nn.dropout(custom_fc_layer_1, keep_prob)

     # Second custom fully connected layer
      custom_fc_weights_2 = weight_variable([1024,2])
      custom_fc_bias_2 = bias_variable([2])
      custom_fc_layer_2 = tf.matmul(custom_fc_layer_1, custom_fc_weights_2) + custom_fc_bias_2

      y_conv = custom_fc_layer_2

      cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
      train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
      correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


      bottleneck_list, y_image_label = convert_images_to_bottlenecks(FLAGS.image_dir)

      le = preprocessing.LabelEncoder()
      y_one_hot = tf.one_hot(le.fit_transform(y_image_label),depth=2)

      x_feed = bottleneck_list

      with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          y_feed = sess.run(y_one_hot)
          x_feed = sess.run(bottleneck_list)
          for i in range(75):

             if i % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                     bottleneck_layer:x_feed, y_: y_feed, keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))

             train_step.run(feed_dict={bottleneck_layer:x_feed , y_: y_feed, keep_prob: 0.8})

      predicted = tf.argmax(y_conv, 1)
      if FLAGS.predict_image != "":
          with tf.Session() as sess:
              x_single_img = sess.run(convert_single_image_to_bottlenecks(FLAGS.predict_image))
              print('You got %s'%le.inverse_transform(sess.run(predicted,feed_dict={x:x_single_img}))[0])






if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--model_dir',
      type=str,
      default='model',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--image_dir',
      type=str,
      default='',
      help='Absolute path to image directory.'
  )
  parser.add_argument(
         '--predict_image',
         type=str,
         default="",
         help='Unknown image'
     )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
