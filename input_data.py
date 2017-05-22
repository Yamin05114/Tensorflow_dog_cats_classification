import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


########## get the image list and label list ###########################################################################
def get_files(file_dir):
    image = []
    label = []
    cat_number = 0
    dog_number = 0
    for file in os.listdir(file_dir):
        image.append(file_dir+file)
        if file.split(sep='.')[0] == 'cat':
            label.append(0)
            cat_number += 1
        else:
            label.append(1)
            dog_number += 1
    print('%d cats and %d dogs' %(cat_number, dog_number))
    if 0.75 <= (cat_number / dog_number) <= 1.2:
        print('The data is well split')
    temp = np.array([image, label])
    temp = temp.transpose()  # the numpy is a mixtype so have to convert label to int
    np.random.shuffle(temp)
    image = list(temp[:, 0])
    label = list(temp[:, 1])
    label = [int(i) for i in label]
    return image, label

'''
image, label = get_files('./data/train')
print(image[0])
print(image[1])
print(label[0])
print(label[1])
'''


########## get the batches #############################################################################################
def get_batches(images, labels, image_w, image_h, batch_size, capacity):
    '''
    :param images: the image list 
    :param labels: the corresponding label
    :param image_w: image width
    :param image_h: image height
    :param batch_size: image number in a batch
    :param capacity: the maximum element in queue
    :return: batch_images: 4D tensor [batch_size, image_height, image_width, 3], dtype = tf.float32
             batch_labels: 1D tensor [batch_size], dtype = tf.int32
    '''
    images = tf.cast(images, tf.string)
    labels = tf.cast(labels, tf.int32)

    # make input queue:
    input_queue = tf.train.slice_input_producer([images, labels])
    ###########################################
    # 1. tensorflow queue is for keep give the input flow. Not really a one time function with one output.
    #    the queue is for feeding the tensorflow batch
    # 2. slice_input_producer: to form a queue for list of tensor -> [image, label]
    #    string_input_producer: to form a queue for string tensor -> image
    #    parameters: control epoch number, shuffle and so on.
    ###########################################

    # input queue can be input_queue[1] since the dequeue element is a tensor list.
    label = input_queue[1]
    image_path = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_path, channels=3)
    ##############################################
    # data argumentation is here
    ##############################################
    image = tf.image.resize_image_with_crop_or_pad(image, image_w, image_h)
    image = tf.image.per_image_standardization(image)  # 0-255 -> 0-1

    # construct the batches.
    batch_images, batch_labels = tf.train.batch([image, label], batch_size,
                                                num_threads=64, capacity=capacity)
    # batch_images, batch_labels = tf.train.shuffle_batch([image, label], batch_size, num_threads=64,
    #                                                    capacity=capacity, min_after_dequeue=)

    batch_labels = tf.reshape(batch_labels, [batch_size])  # column tensor

    return batch_images, batch_labels


########## test and show the batches ###################################################################################
'''
BATCH_SIZE = 2
CAPACITY = 256
IMAGE_W = 208
IMAGE_H = 208

train_dir = './data/train/'

image_list, label_list = get_files(train_dir)
batch_images, batch_labels = get_batches(image_list, label_list, IMAGE_W,
                                         IMAGE_H, BATCH_SIZE, CAPACITY)

with tf.Session() as sess:
    iteration = 0  # the max iteration number.
    coord = tf.train.Coordinator()  # current tensor queue position iterator
    threads = tf.train.start_queue_runners(coord=coord)  # tensor queue enqueue dequeue operations

    try:
        while not coord.should_stop() and iteration < 1:
            images, labels = sess.run([batch_images, batch_labels])

            for j in range(BATCH_SIZE):
                print(images[j, :, :, :].shape)
                plt.imshow(images[j, :, :, :])
                plt.show()
            iteration += 1
    except tf.errors.OutOfRangeError:
        print('Batches Done!')
    finally:
        coord.request_stop()
    coord.join(threads)
'''




















