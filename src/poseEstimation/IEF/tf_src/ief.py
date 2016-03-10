import argparse
import time
import tensorflow as tf
import numpy as np
import cv2
import math
from util import *

def main(**kwargs):
    data_root = '../tf_data/'
    view = 'side'
    num_joints = 15
    small_data = 60 # 100 batches

    # Titan X has 12 GB memory, TensorFlow requires user to specify a fraction
    max_gpu_memory = kwargs.get('gpu_mem')
    gpu_memory_frac = kwargs.get('gpu_frac')
    batch_size = kwargs.get('batch_size')
    learning_rate = kwargs.get('lr')
    num_epochs = kwargs.get('n_epochs')
    num_steps = kwargs.get('n_steps')

    # y's are joints in 2D (x, y)
    X_train, y_train, X_val, y_val = load_data(data_root, view, None)
    drop = (X_train.shape[0])%batch_size
    X_train = X_train[:-drop]
    y_train = y_train[:-drop]
    X_val = X_val[:-drop]
    y_val = y_val[:-drop]

    logger.debug('Train X shape: %s', X_train.shape)
    logger.debug('Train y shape: %s', y_train.shape)
    logger.debug('Val X shape: %s', X_val.shape)
    logger.debug('Val y shape: %s', y_val.shape)

    # Parameters
    input_img_size = 224
    dropout_prob = 0.3
    num_coords = 2
    n_outputs = num_joints * num_coords # How many regression values to output
    n_train = len(X_train)
    num_channel = num_joints + 1

    # vgg
    x = tf.placeholder('float', [None, input_img_size, input_img_size, num_channel])
    y = tf.placeholder('float', [None, n_outputs])
    dropout = tf.placeholder('float')
    y_hat = vgg19(x, y, dropout_prob, n_outputs, input_img_size, num_channel)

    # linear regression
    # x = tf.placeholder("float", [None, input_img_size*input_img_size*num_channel])
    # y = tf.placeholder("float", [None, n_outputs])
    # W = tf.Variable(tf.random_normal([input_img_size*input_img_size*num_channel, n_outputs], stddev=0.01))
    # b = tf.Variable(tf.zeros([n_outputs]))
    # y_hat = tf.add(tf.matmul(x, W), b)

    #cost = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(y - y_hat, 2), 2)))
    cost = tf.reduce_mean(tf.pow(y - y_hat, 2))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.initialize_all_variables()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_frac)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(init)
    start_time = time.time()

    #y_median = np.median(y_train, axis=0)
    #yt_train = np.ones(y_train.shape) * y_median # initialize using mean pose
    #yt_val = np.ones(y_val.shape) * y_median # initialize using mean pose
    yt_train = np.random.permutation(y_train)
    yt_val = np.random.permutation(y_val)
    #visualizeImg(X_train, yt_train)
    #visualizeImg(X_val, yt_val)

    for t in xrange(num_steps):
        num_batches_train = int(np.ceil(1.0 * n_train / batch_size))
        num_batches_val = int(np.ceil(1.0 * X_val.shape[0] / batch_size))
        for epoch in xrange(num_epochs):
            logger.debug('\n----- step %d, epoch %d -----', t, epoch)
            r_order = range(num_batches_train)
            np.random.shuffle(r_order)
            #print 'Batch order', r_order
            for i, b in enumerate(r_order):
                start_idx = b*batch_size
                end_idx = (b+1)*batch_size
                if end_idx > X_train.shape[0]:
                    continue
                #logger.debug('Training using batch %d', b)
                x_batch, eps_batch = get_batch(X_train, y_train, yt_train, start_idx, end_idx, num_joints)
                #x_batch_flat = x_batch.reshape(batch_size, input_img_size*input_img_size*num_channel)
                eps_batch_flat = eps_batch.reshape(batch_size, n_outputs)
                feed = {x: x_batch, y: eps_batch_flat, dropout: dropout_prob}
                sess.run(optimizer, feed_dict=feed)
                if i == 0:
                    eps_pred_flat = sess.run(y_hat, feed_dict=feed) #* 20  # Get eps prediction
                    eps_pred = eps_pred_flat.reshape(batch_size, num_joints, num_coords)

                    loss = sess.run(cost, feed_dict=feed)  # Get loss
                    current_estimate = yt_train[start_idx:end_idx] + eps_pred # Get error
                    loc_err = error(y_train[start_idx:end_idx], current_estimate) # pixel error per joint

                    num_iteration = epoch * num_batches_train + r_order.index(b)
                    logger.debug("\n[INFO:train] Iteration: %s\nLoss: %f\n" + \
                        "Error: %f\nLearning rate: %f", \
                        str(num_iteration), loss, loc_err, learning_rate)

                    '''
                    dists = getDists(y_train[start_idx:end_idx], current_estimate)
                    logger.debug('5cm accuracy: %f', np.sum(dists[:, i] < 5)/ \
                        float(dists.shape[0]))
                    logger.debug('10cm accuracy: %f', np.sum(dists[:, i] < 10)/ \
                        float(dists.shape[0]))
                    logger.debug('15cm accuracy: %f', np.sum(dists[:, i] < 15)/ \
                        float(dists.shape[0]))
                    '''

                    if epoch % 10 == 0:
                        visualizeImgJointsEps(x_batch[:,:,:,0], yt_train[start_idx:end_idx], eps_pred, str(t)+'_'+str(epoch))

            runValidationSet(sess, x, y, dropout, y_hat, cost, X_val, y_val, yt_val, \
                             batch_size, n_outputs, dropout_prob, start_time, num_joints, num_coords)

        train_eps_pred = run_prediction(sess, num_batches_train, batch_size, X_train, y_train, yt_train, \
                                        x, y, y_hat, dropout, dropout_prob, num_joints, num_coords)
        val_eps_pred = run_prediction(sess, num_batches_val, batch_size, X_val, y_val, yt_val, \
                                      x, y, y_hat, dropout, dropout_prob, num_joints, num_coords)
        yt_train += train_eps_pred
        yt_val += val_eps_pred
        #print yt_train.shape, train_eps_pred.shape, yt_val.shape, val_eps_pred.shape

    saver = tf.train.Saver()
    if not os.path.isdir('models'):
        os.makedirs('models')
    saver.save(sess, 'models/ief_vgg')

def error(estimate, ground_truth):
    return np.mean(np.sqrt(np.sum((estimate-ground_truth)**2, 2)))

def runValidationSet(sess, x, y, dropout, y_hat, cost, X_val, y_val, yt_val, \
                     batch_size, n_outputs, keep_prob, start_time, num_joints, num_coords):
    num_batches = int(math.ceil(1.0 * X_val.shape[0] / batch_size))
    accumulator_err = 0.0
    accumulator_cost = 0.0
    for b in xrange(num_batches):
        start_idx = b*batch_size
        end_idx = (b+1)*batch_size
        if end_idx > X_val.shape[0]:
            continue
        x_batch, eps_batch = get_batch(X_val, y_val, yt_val, start_idx, end_idx, num_joints)
        # x_batch = x_batch.reshape(batch_size, 224*224*16)
        eps_batch = eps_batch.reshape(batch_size, n_outputs)
        feed = {x: x_batch, y: eps_batch, dropout: 1.0}
        eps_pred = sess.run(y_hat, feed_dict=feed) # Get eps prediction
        eps_pred = eps_pred.reshape(batch_size, num_joints, num_coords)
        loss = sess.run(cost, feed_dict=feed)
        current_estimate = yt_val[start_idx:end_idx] + eps_pred # Get error
        loc_err = error(y_val[start_idx:end_idx], current_estimate) # pixel error per joint
        batch_weight = 1.0*(end_idx - start_idx) / X_val.shape[0]
        accumulator_err += batch_weight*loc_err
        accumulator_cost += batch_weight*loss
    elapsed_time = 1.0 * (time.time() - start_time) / 60
    logger.debug('\n[INFO:val] Loss: %f\nError: %f\nElapsed: %f min', \
        accumulator_cost, accumulator_err, elapsed_time)

def run_prediction(sess, num_batches, batch_size, X_all, y_all, yt, x, y, y_hat, \
                   dropout, dropout_prob, num_joints, num_coords):
    prediction = []
    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = (b+1)*batch_size
        if end_idx > X_all.shape[0]:
            continue
        x_batch, eps_batch = get_batch(X_all, y_all, yt, start_idx, end_idx, num_joints)
        # x_batch = x_batch.reshape(batch_size, 224*224*16)
        eps_batch = eps_batch.reshape(batch_size, num_joints * num_coords)
        feed = {x: x_batch, y: eps_batch, dropout: dropout_prob}
        feedback = sess.run(y_hat, feed_dict=feed)
        feedback = np.reshape(feedback, (batch_size, num_joints, num_coords))
        prediction.append(feedback)
    prediction = np.vstack(prediction)
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', nargs='?', default=10, type=int)
    parser.add_argument('--n_steps', nargs='?', default=5, type=int)
    parser.add_argument('--n_epochs', nargs='?', default=5, type=int)
    parser.add_argument('--lr', nargs='?', default=5e-4, type=float)
    parser.add_argument('--gpu_mem', nargs='?', default=12287, type=int)
    parser.add_argument('--gpu_frac', nargs='?', default=0.7, type=float)
    args = parser.parse_args()
    main(**vars(args))
