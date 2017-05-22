import tensorflow as tf
import numpy as np
import os
import skimage.io as io
from tqdm import tqdm  # nice progress bars

'''
1. This file is for constructing the CNN networks.
2. All the layers are under the layer's name scope: for tensorboard and param loading
3. I also added pipeline as a layer
4. Since I will write different network construction function for different networks, 
   it is very important to know tf.nn.xxxx will add parts to neural networks, when you return the network,
   when there is no more tf.nn.xxxx not sure about it. now testing.
'''


########## convolutional layer #########################################################################################
def conv(layer_name, x, out_channels, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_trainable=True):
    '''
    :param layer_name: depend on this we can tuning our parameters
    :param x: layer input, or we say, neurons 
    :param out_channels: kernel number
    :param kernel_size: you should know this
    :param stride: as kernel_size
    :param is_pretrain: decide wether start from tuning
    :return: a conv layer
    '''

    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        # xavier_initializer is a good initializer, normal or uniform distribution.
        # read the paper
        w = tf.get_variable(name='weights', trainable=is_trainable,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(name='biases', trainable=is_trainable, shape=[out_channels],
                            initializer=tf.constant_initializer(0.1))
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x


########## pooling layer ###############################################################################################
def pool(layer_name, x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True):
    '''
    :param layer_name: 
    :param x: 
    :param kernel: 
    :param stride: 
    :param is_max_pool: true for max false for average. 
    :return: 
    '''
    with tf.name_scope(layer_name):
        if is_max_pool:
            x = tf.nn.max_pool(x, kernel, strides=stride,
                               padding='SAME', name=layer_name)
        else:
            x = tf.nn.avg_pool(x, kernel, strides=stride,
                               padding='SAME', name=layer_name)
        return x


########## lrn + pool layer ############################################################################################
def lrn_pool(layer_name, x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True):
    with tf.variable_scope(layer_name):
        x = tf.nn.lrn(x, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                      beta=0.75, name='norm')
        if is_max_pool:
            x = tf.nn.max_pool(x, kernel, strides=stride,
                               padding='SAME', name=layer_name)
        else:
            x = tf.nn.avg_pool(x, kernel, strides=stride,
                               padding='SAME', name=layer_name)
        return x


########## batch normalization layer ###################################################################################
def batch_norm(layer_name, x):
    with tf.name_scope(layer_name):
        epsilon = 1e-3
        batch_mean, batch_var = tf.nn.moments(x, [0])
        x = tf.nn.batch_normalization(x, mean=batch_mean, variance=batch_var,
                                      offset=None, scale=None, variance_epsilon=epsilon,
                                      name=layer_name)
        return x


########## fully connected layer #######################################################################################
def FC_layer(layer_name, x, out_nodes):
    '''
    :param layer_name: 
    :param x: 
    :param out_nodes: 
    :return: 
    '''
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights', shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases', shape=[out_nodes],
                            initializer=tf.constant_initializer(0.1))
        flat_x = tf.reshape(x, [-1, size])  # 1d, -1 is for batch size.

        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)
        # print(x.get_shape())
        return x


########## Softmax_linear ##############################################################################################
def softmax_linear(layer_name, x, n_classes):
    '''
    The soft max layer also do a linear transform w * x + b. No relu because softmax is unlinear transform already
    it is not a layer, so can not add to nn.
    :param layer_name: 
    :param x: fc layer output
    :param n_classes: the final classes
    :return: 
    '''
    shape = x.get_shape()
    size = shape[-1].value
    with tf.variable_scope(layer_name) as scope:
        w = tf.get_variable('weights', shape=[size, n_classes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases', shape=[n_classes],
                            initializer=tf.constant_initializer(0.1))
        x = tf.add(tf.matmul(x, w), b, name='softmax_linear')
        return x


########## loss layer ##################################################################################################
def loss(logits, labels):
    '''
    :param logits: logistic regression
    :param labels: desired distribution
    :return: 
    '''
    with tf.name_scope('loss') as scope:
        print(logits.get_shape())
        print(labels.get_shape())
        # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels, name='loss')
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope + '/loss', loss)
        return loss


########## calculate accuracy ##########################################################################################
def accuracy(logits, labels):
    with tf.name_scope('accuracy') as scope:
        # correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct) * 100.0
        tf.summary.scalar(scope + 'accuracy', accuracy)
        return accuracy


########## optimize ####################################################################################################
def optimize(loss, learning_rate):
    '''
    SGD as as a default.
    :param loss: 
    :param learning_rate: 
    :param global_step: 
    :return: 
    '''
    with tf.name_scope('optimize'):
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)  # When continue training, change this.
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op


########## load layer weights and biases ###############################################################################
def load_network_parameter_with_skip(data_path, session, skip_layers):
    # data_path = './VGG16_parameters/vgg16.npy'
    data_dict = np.load(data_path, encoding='latin1').item()
    keys = sorted(data_dict.keys())
    for key in keys:
        if key not in skip_layers:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    session.run(tf.get_variable(subkey).assign(data))
                    # print('\n')
                    # print('weights shape: ', weights.shape)
                    # print('biases shape: ', biases.shape)


########## tfrecord input pipeline #####################################################################################
'''
This is how tfrecord is constructed:
        |-example1
tfrecord|-example2
        |           
        |           
        |         |int64_feature for the label
        |-example3|bytes_feature for images
'''


def int64_feature(value):
    '''
    :param value: usually labels
    :return: the label as a tf.train.Feature 
    '''
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    '''
    :param value: usually images
    :return: the image as a tf.train.Feature 
    '''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def images2tfrecord(images, labels, save_dir, name):
    '''
    conver all the images to 1 tfrecord file
    :param images: a list of strings of image names
    :param labels: a list of int for labels
    :param save_dir: the directory for save the tfrecord label.
    :param name: the name of the tfrecord
    :return: no return
    '''

    filename = os.path.join(save_dir, name + '.tfrecords')
    n_images = len(labels)

    if np.shape(images)[0] != n_images:
        raise ValueError('Image and label number not matched')

    writer = tf.python_io.TFRecordWriter(filename)
    print('\nConstructing TFrecord from the images, it may take time...')

    for i in tqdm(range(0, n_images)):
        try:
            image = io.imread(images[i])  # must be np array
            image = image[0:32, 0:32, :]
            # print(image.shape)
            image_raw = image.tostring()
            label = int(labels[i])
            print(label)

            example = tf.train.Example(features=tf.train.Features(feature={
                'label': int64_feature(label),
                'image_raw': bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())

        except IOError as e:
            print('Can not read: ', images[i])
            print('error: ' %e)
            print('Skip the unreadable image!\n')
    writer.close()

    print('TFrecord done!')


def tfrecord_decode(tfrecords_file, batch_size):
    """
    :param tfrecords_file: a list!!! of tfrecordfile names 
    :param batch_size: 
    :return: images[batch_size, width, height, channel]
             label[batchsize]
             both tensors
    """

    # tf.queue for tfrecord file
    filename_queue = tf.train.string_input_producer(tfrecords_file)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # print(serialized_example)  # test if image is loaded right
    img_features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                           })
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    #  print(image.get_shape().as_list())  # test if image is loaded right
    #####################################################
    # put your data augmentation here if needed
    #####################################################
    # remember to change the image size if needed

    image = tf.reshape(image, [32, 32, 3])
    label = tf.cast(img_features['label'], tf.int32)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size,
                                              num_threads=64,
                                              capacity=2000)
    return image_batch, tf.reshape(label_batch, [batch_size])














