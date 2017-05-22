import os
import tensorflow as tf
import numpy as np
import input_data
import CNN
import Layers

N_CLASSES = 2
IMG_W = 208
IMG_H = 208
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 15000  # epoch * data_size / batch_size
learning_rate = 0.0001  # ???

TRAIN_DIR = './data/train/'
LOGS_TRAIN_DIR = './logs/train/'


def training():

    train_images, train_labels = input_data.get_files(TRAIN_DIR)
    batch_images, batch_labels = input_data.get_batches(train_images, train_labels, IMG_W, IMG_H,
                                                        BATCH_SIZE, CAPACITY)
    train_logits = CNN.CNN(batch_images, BATCH_SIZE, N_CLASSES)
    train_loss = Layers.loss(train_logits, batch_labels)
    train_op = Layers.optimize(train_loss, learning_rate)
    train_accuracy = Layers.accuracy(train_logits, batch_labels)

    summary_op = tf.summary.merge_all()

    sess = tf.Session()
    train_writer = tf.summary.FileWriter(LOGS_TRAIN_DIR, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_accuracy])

            if step % 50 == 0:
                print('Step %d, the training loss is %.2f, train accuracy is %.2f%%' %(step, tra_loss, tra_acc))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 2000 == 0:
                checkpoint_path = os.path.join(LOGS_TRAIN_DIR, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Training Done.')

    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


