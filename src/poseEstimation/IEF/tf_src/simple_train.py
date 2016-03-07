import argparse
import time
import tensorflow as tf
import numpy as np
import cv2
import math
from util import *

def main(**kwargs):
    view = 'side'
    num_joints = 15
    small_data = 60 # 6 batches

    print 'Here!'
    X_train = np.load('/mnt0/emma/IEF/tf_data/depth_' + view + '_train.npy')[:small_data] / 1000
    y_train = np.load('/mnt0/emma/IEF/tf_data/joint_' + view + '_train.npy')[:small_data, :, 2:] / 1000

    X_val = np.load('/mnt0/emma/IEF/tf_data/depth_' + view + '_val.npy')[:small_data/2] / 1000
    y_val = np.load('/mnt0/emma/IEF/tf_data/joint_' + view + '_val.npy')[:small_data/2, :, 2:] / 1000

    col = [1, 2, 0]
    y_train = y_train[:, :, col]
    y_val = y_val[:, :, col]

    print 'Data loaded!'

    print 'Train X shape', X_train.shape
    print 'Train y shape', y_train.shape
    print 'Val X shape', X_val.shape
    print 'Val y shape', y_val.shape
    print 'X max', np.amax(X_train), 'X min', np.amin(X_train)

    # Parameters
    learning_rate = 5e-4
    num_epochs = 1000
    num_iteration = 10
    # Default batch size of 10
    batch_size = 10
    # Titan X has 12 GB memory, TensorFlow requires user to specify a fraction
    max_gpu_memory = 12287  # Need to adjust this for different GPUs! (in MB)
    # gpu_memory_frac = kwargs.get('gpu_memory', 0.9*max_gpu_memory)*1024/max_gpu_memory
    gpu_memory_frac = 0.5
    print gpu_memory_frac
    input_img_size = 224
    dropout_prob = 0.5
    n_outputs =  num_joints * 3 # How many regression values to output
    n_train =len(X_train)

    # x = tf.placeholder("float", [None, input_img_size*input_img_size])
    # y = tf.placeholder("float", [None, n_outputs])
    # W = tf.Variable(tf.zeros([input_img_size*input_img_size, n_outputs]))
    # b = tf.Variable(tf.zeros([n_outputs]))
    # y_hat = tf.add(tf.matmul(x, W), b)

    x = tf.placeholder('float', [None, input_img_size, input_img_size])
    y = tf.placeholder('float', [None, n_outputs])
    dropout = tf.placeholder('float')
    y_hat = vgg19(x, y, dropout_prob, n_outputs, input_img_size, num_joints+1)

    with tf.name_scope("xent") as scope:
        cost = tf.reduce_mean(tf.pow(y - y_hat, 2))/(2*batch_size)
        error = tf.reduce_mean(tf.abs(y - y_hat))
        cost_summ = tf.scalar_summary("L2 Loss", cost)
        error_summ = tf.scalar_summary('Loc Error', error)
    with tf.name_scope("train") as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.initialize_all_variables()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_frac)

    # sess = tf.Session()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(init)
    start_time = time.time()

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("logs/vgg19_log", sess.graph_def)

    saver = tf.train.Saver()
    saver.save(sess, 'regression_vgg', global_step=6000)

    for epoch in xrange(num_epochs):
        num_batches = int(np.ceil(1.0 * n_train / batch_size))
        r_order = range(num_batches)
        np.random.shuffle(r_order)
        print 'Batch order', r_order
        for b in r_order:
            start_idx = b * batch_size
            end_idx = min(X_train.shape[0], (b+1)*batch_size)
            print 'Epoch', epoch, 'Training using batch', b
            # x_batch = X_train[start_idx:end_idx].reshape(batch_size, input_img_size*input_img_size)
            x_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx].reshape(batch_size, n_outputs)
            feed = {x: x_batch, y: y_batch, dropout: dropout_prob}
            sess.run(optimizer, feed_dict=feed)
            if r_order.index(b) % 2 == 0:
                num_iteration = epoch * num_batches + r_order.index(b)
                err = sess.run(error, feed_dict=feed)  # Get error
                loss = sess.run(cost, feed_dict=feed)  # Get loss
                print "[INFO:train] Iteration: " + str(num_iteration) + "\t Loss: " + \
                      "{:.2e}".format(loss) + "\tError: " + "{:.2e}".format(err) + \
                      "\tLearning rate:" + "{:.2e}".format(learning_rate)
                result = sess.run([merged], feed_dict=feed)
                summary_str = result[0]
                writer.add_summary(summary_str, num_iteration)
        runValidationSet(sess, x, y, dropout, y_hat, error, cost, X_val, y_val, \
                         batch_size, n_outputs, dropout_prob, start_time)

    prediction = []
    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min(X_train.shape[0], (b+1)*batch_size)
        # x_batch = X_train[start_idx:end_idx].reshape(batch_size, input_img_size*input_img_size)
        x_batch = X_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx].reshape(batch_size, n_outputs)
        feed = {x: x_batch, y: y_batch, dropout: dropout_prob}
        feedback = sess.run(y_hat, feed_dict=feed)
        feedback = np.reshape(feedback, (batch_size, num_joints, 3))
        prediction.append(feedback)
    prediction = np.vstack(prediction)
    np.save('small_data_pred_vgg.npy', prediction)

def runValidationSet(sess, x, y, dropout, y_hat, error, cost, X_val, y_val, \
                     batch_size, n_outputs, keep_prob, start_time):
    num_batches = int(math.ceil(1.0 * X_val.shape[0] / batch_size))
    accumulator_err = 0.0
    accumulator_cost = 0.0
    for b in xrange(num_batches):
        start_idx = b * batch_size
        end_idx = min(X_val.shape[0], (b+1)*batch_size)
        x_batch = X_val[start_idx:end_idx]
        # x_batch = X_val[start_idx:end_idx].reshape(batch_size, input_img_size*input_img_size)
        y_batch = y_val[start_idx:end_idx].reshape(batch_size, n_outputs)
        feed = {x: x_batch, y: y_batch, dropout: 1.0}
        err = sess.run(error, feed_dict=feed)
        loss = sess.run(cost, feed_dict=feed)
        batch_weight = 1.0*(end_idx - start_idx) / X_val.shape[0]
        accumulator_err += batch_weight*err
        accumulator_cost += batch_weight*loss
    elapsed_time = 1.0 * (time.time() - start_time) / 60
    print '[INFO:val] Loss: %f\t Error: %f\t Elapsed: %0.1f min' % (accumulator_cost, accumulator_err, elapsed_time)

def vanillaNet(x, keep_prob, n_outputs, input_img_size):
    x_image = tf.reshape(x, [-1, input_img_size, input_img_size, 1])
    W = {}
    b = {}

    W['c1'] = weight_variable([5, 5, 1, 32])
    b['c1'] = bias_variable([32])

    conv1 = tf.nn.relu(conv2d(x_image, W['c1']) + b['c1'])
    pool1 = max_pool_2x2(conv1)

    W['c2'] = weight_variable([5, 5, 32, 64])
    b['c2'] = bias_variable([64])

    conv2 = tf.nn.relu(conv2d(pool1, W['c2']) + b['c2'])
    pool2 = max_pool_2x2(conv2)

    W['fc1'] = weight_variable([56 * 56 * 64, 1024])
    b['fc1'] = bias_variable([1024])

    h_pool2_flat = tf.reshape(pool2, [-1, 56*56*64])
    h_fc1 = tf.add(tf.matmul(h_pool2_flat, W['fc1']), b['fc1'])

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W['fc2'] = weight_variable([1024, n_outputs])
    b['fc2'] = bias_variable([n_outputs])

    y_hat = tf.add(tf.matmul(h_fc1_drop, W['fc2']), b['fc2'])

    return y_hat

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # GPU memory to use, in GB
    # parser.add_argument('--gpu_memory', type=float, required=True)
    # Batch size to use
    parser.add_argument('--batch_size', type=int, required=True)
    args = parser.parse_args()
    main(**vars(args))
