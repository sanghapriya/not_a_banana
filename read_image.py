import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import os

def prepare_data(img_dir):

    image_dirs = np.array([dirpath for (dirpath, dirnames, filenames) in gfile.Walk(os.getcwd()+'/'+img_dir)])
    file_list = []
    y_= []
# Ignoring the first directory as it is the base directory
    for image_dir in image_dirs[1:]:
            extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
            dir_name = os.path.basename(image_dir)
            image_file_list =[]
            tf.logging.info("Looking for images in '" + dir_name + "'")
            for extension in extensions:
                # Building the filename pattern
                file_glob = os.path.join(image_dir,'*.' + extension)
                #This looks for a file name pattern, helps us ensure that only jpg extensions are choosen
                image_file_list = gfile.Glob(file_glob)
                file_list.extend(image_file_list)
                y_.extend([dir_name]*len(image_file_list))

    return file_list,y_



def read_image_array(image_loc_array):
    resized_image_array=[]

    for image_loc in image_loc_array:
        image_decoded = tf.image.decode_jpeg(tf.gfile.FastGFile(image_loc, 'rb').read(),channels=3)
        resized_image = tf.reshape(tf.image.resize_images(image_decoded, [28,28]),[1,28*28*3])
        resized_image_array.append(resized_image)

    resized_image_array = tf.reshape(tf.stack(resized_image_array),[len(image_loc_array),28*28*3])
    return resized_image_array





def read_single_image(image_loc):
    image_decoded = tf.image.decode_jpeg(tf.gfile.FastGFile(image_loc, 'rb').read(),channels=3)
    resized_image = tf.reshape(tf.image.resize_images(image_decoded, [28,28]),[1,28*28*3])
    return resized_image
