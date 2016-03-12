import argparse
import time
import numpy as np
import cv2
import math
from util import *

def main(**kwargs):
    #data_root = '../tf_data/'
    data_root = kwargs.get('indir')
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
    load_model = kwargs.get('load_model')

    # y's are joints in 2D (x, y)
    X_train, y_train, X_val, y_val = load_data(data_root, view, None)
    # print np.mean(X_train[0]), np.mean(X_train[0][X_train[0] != 0])
    # print y_train[0]
    # return

    drop = (X_train.shape[0]) % batch_size
    print 'drop: %d' % drop
    if drop != 0:
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
    n_train, n_val = len(X_train), len(X_val)
    num_channel = num_joints + 1

    hm = create_single_heatmap(input_img_size, input_img_size)

    # vgg
    x = tf.placeholder('float', [None, input_img_size, input_img_size, num_channel])
    y = tf.placeholder('float', [None, n_outputs])
    dropout = tf.placeholder('float')
    y_hat = vgg19(x, y, dropout_prob, n_outputs, input_img_size, num_channel)

    # linear regression
    # x = tf.placeholder('float', [None, input_img_size*input_img_size*num_channel])
    # y = tf.placeholder('float', [None, n_outputs])
    # W = tf.Variable(tf.random_normal([input_img_size*input_img_size*num_channel, n_outputs], stddev=0.01))
    # b = tf.Variable(tf.zeros([n_outputs]))
    # y_hat = tf.add(tf.matmul(x, W), b)

    #cost = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(y - y_hat, 2), 2)))
    cost = tf.reduce_mean(tf.pow(y - y_hat, 2))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    saver = tf.train.Saver(tf.all_variables())

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_frac)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    if load_model:
        saver.restore(sess, 'models/ief.ckpt')
    else:
        sess.run(tf.initialize_all_variables())

    start_time = time.time()

    #y_median = np.median(y_train, axis=0)
    #yt_train = np.ones(y_train.shape) * y_median # initialize using mean pose
    #yt_val = np.ones(y_val.shape) * y_median # initialize using mean pose
    yt_train = np.random.permutation(y_train)
    yt_val = np.random.permutation(y_val)
    #visualizeImg(X_train, yt_train)
    #visualizeImg(X_val, yt_val)

    if not os.path.isdir('models'):
        os.makedirs('models')

    for t in xrange(num_steps):
        num_batches_train = int(np.ceil(1.0*n_train/batch_size))
        num_batches_val = int(np.ceil(1.0*n_val/batch_size))
        for epoch in xrange(num_epochs):
            logger.debug('\n\n----- step %d, epoch %d -----', t, epoch)
            r_order = range(num_batches_train)
            np.random.shuffle(r_order)
            for i, b in enumerate(r_order):
                start_idx = b*batch_size
                end_idx = min(n_train, (b+1)*batch_size)
                x_batch, eps_batch = get_batch(X_train, y_train, \
                    yt_train, start_idx, end_idx, num_joints, hm)
                eps_batch_flat = eps_batch.reshape(batch_size, n_outputs)
                print 'mean eps_batch: %f' % np.mean(np.abs(eps_batch_flat))
                # feed = {x: x_batch, y: eps_batch_flat, dropout: 1.0}
                feed = {x: x_batch, y: eps_batch_flat}
                sess.run(optimizer, feed_dict=feed)

                if i == 0:
                    #if epoch % 5 == 0:
                    num_iteration = epoch*num_batches_train+r_order.index(b)
                    saver.save(sess, 'models/ief.ckpt')

                    eps_pred_flat = sess.run(y_hat, feed_dict=feed) # Get eps prediction
                    print 'mean eps_pred: %f' % np.mean(np.abs(eps_pred_flat))
                    eps_pred = eps_pred_flat.reshape(batch_size, num_joints, num_coords)
                    loss = sess.run(cost, feed_dict=feed)  # Get loss
                    current_estimate = yt_train[start_idx:end_idx] + eps_pred # Get error
                    loc_err = error(y_train[start_idx:end_idx], current_estimate) # pixel error per joint
                    num_iteration = epoch*num_batches_train+r_order.index(b)
                    logger.debug('\n[INFO:train] Iteration: %s\nLoss: %f\n' + \
                        'Error: %f\nLearning rate: %f', \
                        str(num_iteration), loss, loc_err, learning_rate)

                    dists = get_distances(X_train[start_idx:end_idx], y_train[start_idx:end_idx], current_estimate)*100
                    logger.debug('average distance: %f cm', np.mean(dists))
                    logger.debug('5cm accuracy: %f', np.sum(dists < 5)/float(dists.shape[0]*dists.shape[1]))
                    logger.debug('10cm accuracy: %f', np.sum(dists < 10)/ \
                        float(dists.shape[0]*dists.shape[1]))
                    logger.debug('15cm accuracy: %f', np.sum(dists < 15)/ \
                        float(dists.shape[0]*dists.shape[1]))

                    if epoch % 10 == 0:
                        visualizeImgJointsEps(x_batch[:,:,:,0], \
                            yt_train[start_idx:end_idx], eps_pred, \
                            name=str(t)+'_'+str(epoch))

            run_validation(sess, x, y, dropout, y_hat, cost, X_val, y_val, yt_val, \
                             batch_size, n_outputs, dropout_prob, start_time, num_joints, num_coords, hm)

        train_eps_pred = run_prediction(sess, num_batches_train, batch_size, X_train, y_train, yt_train, \
                                        x, y, y_hat, dropout, dropout_prob, num_joints, num_coords, hm)
        val_eps_pred = run_prediction(sess, num_batches_val, batch_size, X_val, y_val, yt_val, \
                                      x, y, y_hat, dropout, dropout_prob, num_joints, num_coords, hm)
        yt_train += train_eps_pred
        yt_val += val_eps_pred
        print 'mean train_eps_pred: %f' % np.mean(np.abs(train_eps_pred))
        print 'mean val_eps_pred: %f' % np.mean(np.abs(val_eps_pred))
        #print yt_train.shape, train_eps_pred.shape, yt_val.shape, val_eps_pred.shape

def error(estimate, ground_truth):
    return np.mean(np.sqrt(np.sum((estimate-ground_truth)**2, 2)))

def run_validation(sess, x, y, dropout, y_hat, cost, X_val, y_val, yt_val, \
                     batch_size, n_outputs, keep_prob, start_time, num_joints, num_coords, hm):
    num_batches = int(math.ceil(1.0*X_val.shape[0]/batch_size))
    accumulator_err = 0.0
    accumulator_cost = 0.0
    for b in xrange(num_batches):
        start_idx = b*batch_size
        end_idx = min(X_val.shape[0], (b+1)*batch_size)
        x_batch, eps_batch = get_batch(X_val, y_val, yt_val, start_idx, end_idx, num_joints, hm)
        # x_batch = x_batch.reshape(batch_size, 224*224*16)
        eps_batch = eps_batch.reshape(batch_size, n_outputs)
        # feed = {x: x_batch, y: eps_batch, dropout: 1.0}
        feed = {x: x_batch, y: eps_batch}
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
                   dropout, dropout_prob, num_joints, num_coords, hm, L=20):
    prediction = []
    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min(X_all.shape[0], (b+1)*batch_size)
        x_batch, eps_batch = get_batch(X_all, y_all, yt, start_idx, end_idx, num_joints, hm)
        # x_batch = x_batch.reshape(batch_size, 224*224*16)
        eps_batch = eps_batch.reshape(batch_size, num_joints*num_coords)
        # feed = {x: x_batch, y: eps_batch, dropout: 1.0}
        feed = {x: x_batch, y: eps_batch}
        feedback = sess.run(y_hat, feed_dict=feed)
        feedback = np.reshape(feedback, (batch_size, num_joints, num_coords))
        prediction.append(feedback)
    prediction = np.vstack(prediction)
    prediction = np.clip(prediction, -L, L)
    return prediction

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', nargs='?', default=10, type=int)
    parser.add_argument('--n_steps', nargs='?', default=10, type=int)
    parser.add_argument('--n_epochs', nargs='?', default=50, type=int)
    parser.add_argument('--lr', nargs='?', default=5e-4, type=float)
    parser.add_argument('--gpu_mem', nargs='?', default=12287, type=int)
    parser.add_argument('--gpu_frac', nargs='?', default=0.5, type=float)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--indir')
    args = parser.parse_args()
    main(**vars(args))
