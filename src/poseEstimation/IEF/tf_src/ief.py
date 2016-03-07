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
    offset = 115

    # y's are joints in 2D (x, y)
    X_train, y_train, X_val, y_val = load_data(data_root, view, small_data, offset)

    print 'Train X shape', X_train.shape
    print 'Train y shape', y_train.shape
    print 'Val X shape', X_val.shape
    print 'Val y shape', y_val.shape

    # Parameters
    learning_rate = 1e-3
    num_epochs = 200
    num_step = 5
    # Default batch size of 10
    batch_size = 10
    # Titan X has 12 GB memory, TensorFlow requires user to specify a fraction
    max_gpu_memory = 2048 #12287  # Need to adjust this for different GPUs! (in MB)
    # gpu_memory_frac = kwargs.get('gpu_memory', 0.9*max_gpu_memory)*1024/max_gpu_memory
    gpu_memory_frac = 0.5
    input_img_size = 224
    dropout_prob = 0.3
    num_coords = 2
    n_outputs = num_joints * num_coords # How many regression values to output
    n_train = len(X_train)
    num_channel = num_joints + 1

    # x = tf.placeholder('float', [None, input_img_size, input_img_size, num_channel])
    # y = tf.placeholder('float', [None, n_outputs])
    dropout = tf.placeholder('float')
    # y_hat = vgg19(x, y, dropout_prob, n_outputs, input_img_size, num_channel)

    x = tf.placeholder("float", [None, input_img_size*input_img_size*num_channel])
    y = tf.placeholder("float", [None, n_outputs])
    W = tf.Variable(tf.random_normal([input_img_size*input_img_size*num_channel, n_outputs], stddev=0.01))
    b = tf.Variable(tf.zeros([n_outputs]))
    y_hat = tf.add(tf.matmul(x, W), b)

    cost = tf.reduce_mean(tf.pow(y - y_hat, 2))
    error = tf.reduce_mean(tf.abs(y - y_hat))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.initialize_all_variables()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_frac)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(init)
    start_time = time.time()

    saver = tf.train.Saver()
    if not os.path.isdir('models'):
        os.makedirs('models')
    saver.save(sess, 'models/ief_vgg', global_step=5000)

    y_median = np.median(y_train, axis=0)
    yt_train = np.ones(y_train.shape) * y_median # initialize using mean pose
    yt_val = np.ones(y_val.shape) * y_median # initialize using mean pose

    #visualizeImg(X_train, yt_train)
    #visualizeImg(X_val, yt_val)

    for t in xrange(num_step):
        num_batches_train = int(np.ceil(1.0 * n_train / batch_size))
        num_batches_val = int(np.ceil(1.0 * X_val.shape[0] / batch_size))
        for epoch in xrange(num_epochs):
            print '----- step %d, epoch %d -----' % (t, epoch)
            r_order = range(num_batches_train)
            np.random.shuffle(r_order)
            print 'Batch order', r_order
            for b in r_order:
                start_idx = b * batch_size
                end_idx = min(X_train.shape[0], (b+1)*batch_size)
                print 'Training using batch %d' % b
                x_batch, eps_batch = get_batch(X_train, y_train, yt_train, start_idx, end_idx, num_joints)
                x_batch_flat = x_batch.reshape(batch_size, input_img_size*input_img_size*num_channel)
                eps_batch_flat = eps_batch.reshape(batch_size, n_outputs) #/ 20.0 # normalize labels to be -1 to 1
                #print 'Max x:', np.amax(x_batch), 'Min x:', np.amin(x_batch)
                #print 'Max correction:', np.amax(eps_batch), 'Min correction:', np.amin(eps_batch)
                #feed = {x: x_batch_flat, y: eps_batch_flat, dropout: dropout_prob}
                feed = {x: x_batch_flat, y: eps_batch_flat}
                sess.run(optimizer, feed_dict=feed)
                if b == 0:
                    eps_pred_flat = sess.run(y_hat, feed_dict=feed) #* 20  # Get eps prediction
                    eps_pred = eps_pred_flat.reshape(batch_size, num_joints, num_coords)
                    loss = sess.run(cost, feed_dict=feed)  # Get loss
                    current_estimate = yt_train[start_idx:end_idx] + eps_pred # Get error
                    loc_err = np.mean(np.sqrt(np.sum((y_train[start_idx:end_idx]-current_estimate)**2, 2))) # pixel error per joint
                    num_iteration = epoch * num_batches_train + r_order.index(b)
                    print "\n[INFO:train] Iteration: " + str(num_iteration) + "\nLoss: " + \
                          "{:.2f}".format(loss) + "\nError: " + "{:.2f}".format(loc_err) + \
                          "\nLearning rate: " + "{:.7f}".format(learning_rate) + '\n'
                    if epoch % 10 == 0:
                        visualizeImgJointsEps(x_batch[:,:,:,0], yt_train[start_idx:end_idx], eps_pred)

            runValidationSet(sess, x, y, dropout, y_hat, error, cost, X_val, y_val, yt_val, \
                             batch_size, n_outputs, dropout_prob, start_time, num_joints, num_coords)
        train_eps_pred = run_prediction(sess, num_batches_train, batch_size, X_train, y_train, yt_train, \
                                        x, y, y_hat, dropout, dropout_prob, num_joints, num_coords)
        val_eps_pred = run_prediction(sess, num_batches_val, batch_size, X_val, y_val, yt_val, \
                                      x, y, y_hat, dropout, dropout_prob, num_joints, num_coords)
        yt_train += train_eps_pred
        yt_val += val_eps_pred


        np.save('small_data_pred_vgg.npy', yt_train)
        np.save('small_data_pred_vgg_val.npy', yt_val)

def runValidationSet(sess, x, y, dropout, y_hat, error, cost, X_val, y_val, yt_val, \
                     batch_size, n_outputs, keep_prob, start_time, num_joints, num_coords):
    num_batches = int(math.ceil(1.0 * X_val.shape[0] / batch_size))
    accumulator_err = 0.0
    accumulator_cost = 0.0
    for b in xrange(num_batches):
        start_idx = b * batch_size
        end_idx = min(X_val.shape[0], (b+1)*batch_size)
        x_batch, eps_batch = get_batch(X_val, y_val, yt_val, start_idx, end_idx, num_joints)
        x_batch = x_batch.reshape(batch_size, 224*224*16)
        eps_batch = eps_batch.reshape(batch_size, n_outputs) / 20
        feed = {x: x_batch, y: eps_batch, dropout: 1.0}
        eps_pred = sess.run(y_hat, feed_dict=feed) * 20 # Get eps prediction
        eps_pred = eps_pred.reshape(batch_size, num_joints, num_coords)
        loss = sess.run(cost, feed_dict=feed)
        current_estimate = yt_val[start_idx:end_idx] + eps_pred # Get error
        loc_err = np.mean(np.abs(y_val[start_idx:end_idx] - current_estimate))
        batch_weight = 1.0*(end_idx - start_idx) / X_val.shape[0]
        accumulator_err += batch_weight*loc_err
        accumulator_cost += batch_weight*loss
    elapsed_time = 1.0 * (time.time() - start_time) / 60
    print '[INFO:val] Loss: %f\t Error: %f\t Elapsed: %0.1f min' % (accumulator_cost, accumulator_err, elapsed_time)

def run_prediction(sess, num_batches, batch_size, X_all, y_all, yt, x, y, y_hat, \
                   dropout, dropout_prob, num_joints, num_coords):
    prediction = []
    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min(X_all.shape[0], (b+1)*batch_size)
        x_batch, eps_batch = get_batch(X_all, y_all, yt, start_idx, end_idx, num_joints)
        x_batch = x_batch.reshape(batch_size, 224*224*16)
        eps_batch = eps_batch.reshape(batch_size, num_joints * num_coords)
        feed = {x: x_batch, y: eps_batch, dropout: dropout_prob}
        feedback = sess.run(y_hat, feed_dict=feed)
        feedback = np.reshape(feedback, (batch_size, num_joints, num_coords))
        prediction.append(feedback)
    prediction = np.vstack(prediction)
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # GPU memory to use, in GB
    # parser.add_argument('--gpu_memory', type=float, required=True)
    # Batch size to use
    parser.add_argument('--batch_size', type=int, required=True)
    args = parser.parse_args()
    main(**vars(args))
