'''
This file is for defining the network structure. 
For a CNN, usually, you need the following parameters as input:
batch_size: 32, 64, ... 
batch_images: 4D tensor [batch_size, image_width, image_height, channels]
batch_labels: 1D tensor [batch_size]
etc: ...
'''

import tensorflow as tf
import Layers
import input_data


def CNN(batch_images, batch_labels, batch_size):
    with tf.name_scope('VGG16'):
        # conv-pool 1
        x = Layers.conv('conv1', batch_images, 16, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = Layers.lrn_pool('pool1', x, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1])
        # conv-pool 2
        x = Layers.conv('conv2', x, 16, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = Layers.lrn_pool('pool2', x, kernel=[1, 2, 2, 1], stride=[1, 1, 1, 1])

        # fc-1
        x = Layers.FC_layer('fc3', x, out_nodes=128)
        x = Layers.batch_norm('Batch_norm', x)
        x = Layers.FC_layer('fc4', x, out_nodes=128)
        # softmax
        x = Layers.softmax_linear('Softmax', x, 2)
        return x

