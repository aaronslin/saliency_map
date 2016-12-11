################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
from alexnetcam_params import *

import cPickle
import tensorflow as tf
print ""

def process_image(image_buffer):
    image = tf.image.decode_jpeg(image_buffer, channels=3)
    image = tf.image.resize_images(image, [IMAGE_SIZE, IMAGE_SIZE])
    return image

def process_label(label_buffer):
    label = tf.one_hot(label_buffer, N_CLASSES)
    label = tf.reshape(label, [N_CLASSES])
    return label

def read_and_decode(filenames):
    # filenames is probably supposed to be a list of the shard names
    # TODO: make sure that the images are actually resized!!
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    
    feature_map = {
            "image/encoded": tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            "image/class/label": tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
        }
    features = tf.parse_single_example(
        serialized_example,
        features = feature_map)
    label = features['image/class/label']
    image = features['image/encoded']
    
    image = process_image(image)
    label = process_label(label)
    return image, label

def get_filenames(subset):
    searchPattern = os.path.join(SHARD_DIR, '%s-*' % subset)
    dataFiles = tf.gfile.Glob(searchPattern)
    if not dataFiles:
        print "WARNING: No data files found when searching for shards."
    return dataFiles
filenames = get_filenames("validation")

image, label = read_and_decode(filenames)
print "Image shape:", image.get_shape()
print "Label shape:", label.get_shape()

images_batch, labels_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=BATCH_SIZE,
        capacity=2*BATCH_SIZE,
        min_after_dequeue=BATCH_SIZE)


################################################################################

net_data = load("bvlc_alexnet.npy").item()

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

# TODO: idk what x is actually supposed to do
# ANS: use train_x for x; input should be the placeholder that goes in the feed_dict key
# x = tf.placeholder(tf.float32, (None,) + xdim)

# train_x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
# train_y = tf.placeholder(tf.float32, shape=[None, 1])
#xdim = tuple(train_x.get_shape()[1:])
#ydim = tuple(train_y.get_shape()[1:])

# TODO: can you actually just put the graph inside of a method like this?
def network_model(train_x): 
    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(train_x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)


    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)


    #conv5
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5W = tf.Variable(net_data["conv5"][0])
    conv5b = tf.Variable(net_data["conv5"][1])
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)


    # GAP additions

    # conv6 layer
    k_h = 3; k_w = 3; c_o = N_CLASSES; s_h = 1; s_w = 1; group = 1
    conv6W = tf.Variable(tf.zeros([3,3,256,N_CLASSES]))
    conv6b = tf.Variable(tf.zeros([N_CLASSES]))
    conv6 = conv(conv5,conv6W, conv6b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)


    # GAP layer
    k_h = int(conv6.get_shape()[1]); k_w = int(conv6.get_shape()[2]); s_h = 1; s_w = 1;
    gap_unsqueezed = tf.nn.avg_pool(conv6, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding='VALID')
    gap = tf.squeeze(gap_unsqueezed)


    # fully-connected layer
    fc_newW = tf.Variable(tf.zeros([N_CLASSES,N_CLASSES]))
    fc_newB = tf.Variable(tf.zeros([N_CLASSES]))
    fc_new = tf.nn.xw_plus_b(gap, fc_newW, fc_newB)

    #prob
    #softmax(name='prob'))
    probs = tf.nn.softmax(fc_new)

    return probs

################################################################################

# Loss/Optimizer
# TODO: play around with the learning rate
# TODO: figure out where exactly the accuracy stuff should be returned/evaluated
train_x = images_batch
train_y = labels_batch
probs = network_model(train_x)

print "train_y", train_y.get_shape()
print "probs", probs.get_shape()
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(probs, train_y))
optimizer = tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(loss)

# Evaluate model
correct_pred = tf.equal(tf.argmax(probs,1), tf.argmax(train_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    for stepoch in range(N_EPOCHS):
        # TODO: resize!!
        print ""
        #feed_dict = {train_x: images_batch, train_y: labels_batch}
        print "finished feeddict"
        _, l, predictions = sess.run([optimizer, loss, correct_pred])
        print "finish end of loop"
        print l, predictions
        print ""











################################################################################

#  DEPRECATED

################################################################################

def process_image_old(image):
    imageSize = [IMAGE_SIZE, IMAGE_SIZE, 3]
    if image.ndim==2:
        image = image.reshape(image.shape+(1,))
        image = np.tile(image, 3)
    return imresize(image, imageSize)

def save_image_data(dataName, homedir=HOME_DIR):
    categories = os.listdir(homedir)
    fileNames = [os.listdir(homedir+cat) for cat in categories]
    all_files = [homedir+cat+"/"+file for (cat, files) in zip(categories, fileNames) for file in files]
    labels = [int(cat.strip("category")) for cat in categories]
    all_labels = np.array([label for (label, files) in zip(labels, fileNames) for file in files])

    # # flatten arrays
    all_files = all_files[:1000]
    # all_files = sum(all_files, [])
    # all_labels = sum(all_labels, [])


    print "Loading images..."
    t1 = time.time()
    all_images = np.array([process_image(imread(addr)) for addr in all_files])
    t2 = time.time()
    print "Finished loading images:", t2-t1, "seconds (?); shape:", all_images.shape

    f = open(dataName+"_images.pickle", "wb")
    cPickle.dump(all_images, f)
    f.close()
    g = open(dataName+"_labels.pickle", "wb")
    cPickle.dump(all_labels, g)
    g.close()
    print "Finished saving to:", dataName

    return all_files, all_labels

def load_data(dataName):
    f = open(dataName+"_images.pickle", "rb")
    g = open(dataName+"_labels.pickle", "rb")
    images = cPickle.load(f)
    labels = cPickle.load(g)
    f.close()
    g.close()

    return images, labels

# [1] https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/
# [2] https://github.com/tensorflow/models/blob/master/inception/inception/image_processing.py











