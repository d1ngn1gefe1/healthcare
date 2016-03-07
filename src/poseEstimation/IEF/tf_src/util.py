import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal
import os
import cv2

def save_data(data_root, out_dir, view, person_id_list, d_type):
    depth_view = []
    joint_view = []
    for i in person_id_list:
        index = str(i).zfill(2)
        print 'Loading', data_root + index
        depth = np.load(data_root + index + '_depth_' + view + '.npy')
        joint = np.load(data_root + index + '_joints_' + view + '.npy')
        depth_view.append(depth)
        joint_view.append(joint)
    depth_view = np.vstack(depth_view)
    joint_view = np.vstack(joint_view)
    np.save(out_dir + 'depth_' + view + '_' + d_type + '.npy', depth_view)
    np.save(out_dir + 'joint_' + view + '_' + d_type + '.npy', joint_view)

def resize(data_dir, img_height, img_width):
    depth_people = [data_dir + d for d in os.listdir(data_dir) if d.find('depth') != -1]
    for f in depth_people:
        print 'Loading', f
        images = np.load(f)
        images_new = np.zeros((len(images), img_height, img_width))
        for i in range(len(images)):
            if i % 100 == 0:
                print 'Processed', i, '/', len(images)
            images_new[i] = cv2.resize(images[i], (img_height, img_width))
        np.save(f, images_new)

def load_data(data_root, view, small_data=None, offset=0, overwrite=False):
    X_train, y_train, X_val, y_val = None, None, None, None

    if (small_data is not None) and (not overwrite) and \
        os.path.exists(data_root+'depth_'+view+'_train_small.npy'):
        X_train = np.load(data_root+'depth_'+view+'_train_small.npy')
        y_train = np.load(data_root+'joint_'+view+'_train_small.npy')
        X_val = np.load(data_root+'depth_'+view+'_val_small.npy')
        y_val = np.load(data_root+'joint_'+view+'_val_small.npy')
    elif (not os.path.exists(data_root+'depth_'+view+'_train_small.npy')) or \
        overwrite:
        X_train = np.load(data_root+'depth_'+view+'_train.npy')
        y_train = np.load(data_root+'joint_'+view+'_train.npy')
        X_val = np.load(data_root+'depth_'+view+'_val.npy')
        y_val = np.load(data_root+'joint_'+view+'_val.npy')
        y_train[:, :, 0] *= 224.0/320
        y_train[:, :, 1] *= 224.0/240
        y_val[:, :, 0] *= 224.0/320
        y_val[:, :, 1] *= 224.0/240

        if small_data is not None:
            X_train = X_train[offset:(offset+small_data)] / 1000.0
            # y_train = y_train[offset:offset+small_data, :, 2:] / 1000
            y_train = y_train[offset:(offset+small_data), :, :2] # 2D (x, y)

            # X_train = np.load(data_root + 'small_train_X.npy') / 1000
            # y_train = np.load(data_root + 'small_train_y.npy')

            X_val = X_val[offset:(offset+small_data/2)] / 1000.0
            # y_val = y_val[offset:offset+small_data/2, :, 2:] / 1000
            y_val = y_val[offset:(offset+small_data/2), :, :2]
            np.save(data_root+'depth_'+view+'_train_small.npy', X_train)
            np.save(data_root+'joint_'+view+'_train_small.npy', y_train)
            np.save(data_root+'depth_'+view+'_val_small.npy', X_val)
            np.save(data_root+'joint_'+view+'_val_small.npy', y_val)

    '''
    for i in range(X_train.shape[0]):
        cv2.normalize(X_train[i], X_train[i], 0, 255, cv2.NORM_MINMAX)
        img = cv2.equalizeHist(X_train[i].astype(np.uint8))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for j in range(y_train.shape[1]):
            cv2.circle(img, tuple(y_train[i, j, :2].astype(np.uint16)), 2, (255, 0, 0), -1)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    '''

    # col = [1, 2, 0]
    # y_train = y_train[:, :, col]
    # y_val = y_val[:, :, col]

    return X_train, y_train, X_val, y_val

def visualizeImgJointsEps(imgs, joints=None, eps=None, name='img'):
    for i in range(imgs.shape[0]):
        cv2.normalize(imgs[i], imgs[i], 0, 255, cv2.NORM_MINMAX)
        img = cv2.equalizeHist(imgs[i].astype(np.uint8))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_corrected = img.copy()

        if joints is not None:
            for j in range(joints.shape[1]):
                cv2.circle(img, tuple(joints[i, j].astype(np.uint16)), \
                    2, (255, 0, 0), -1)
        if (joints is not None) and (eps is not None):
            for j in range(joints.shape[1]):
                joints_corrected = joints[i, j] + eps[i, j]
                cv2.circle(img_corrected, \
                    tuple(joints_corrected.astype(np.uint16)), \
                    2, (255, 0, 0), -1)
        cv2.imshow(name, img)
        cv2.imshow(name+'_corrected', img_corrected)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualizeImgHmsEps(x, yt, eps, name='img'):
    # 224 x 224 x 16
    cv2.normalize(x[:, :, 0], x[:, :, 0], 0, 255, cv2.NORM_MINMAX)
    img = cv2.equalizeHist(x[:, :, 0].astype(np.uint8))
    cv2.imshow(name, img)

    hms = np.sum(x[:, :, 1:], axis=2)
    cv2.normalize(hms, hms, 0, 255, cv2.NORM_MINMAX)
    hms = cv2.equalizeHist(hms.astype(np.uint8))
    hms = cv2.cvtColor(hms, cv2.COLOR_GRAY2RGB)
    for i, e in enumerate(eps):
        cv2.line(hms, tuple(yt[i].astype(np.uint16)), \
            tuple((yt[i]+e).astype(np.uint16)), (0, 0, 255), 2)

    cv2.imshow(name+'-hms', hms)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def world2pixel(coordW, W=320, H=240, C=3.51e-3):
    coordP = np.zeros(coordW.shape)
    coordP[:,:,0] = coordW[:,:,0] / (coordW[:,:,2]*C) + 0.5 * W
    coordP[:,:,1] = -(coordW[:,:,1] / (coordW[:,:,2]*C)) + 0.5 * H
    coordP[:,:,2] = coordW[:,:,2]
    return coordP

def joint_to_hm(joints, num_joints, img_height=224, img_width=224):
    cov = [[15, 0], [0, 15]]
    hm_shape = np.ones((img_height, img_width))
    pair = np.nonzero(hm_shape)
    hm_index = np.array(zip(pair[0],pair[1])).reshape(img_height, img_width, 2)
    heat_maps = []
    for i in range(num_joints):
        mean = [joints[i][1], joints[i][0]]
        hm = multivariate_normal.pdf(hm_index, mean, cov)
        scale = np.amax(hm) - np.amin(hm)
        hm /= scale
        heat_maps.append(hm)
    heat_maps = np.array(heat_maps)
    return heat_maps

def add_hms(images, yt, num_joints, num_channel=1, H=240.0, W=320.0):
    out = []
    N, img_height, img_width = images.shape
    # joints = world2pixel(yt) # joint in 240 x 320 pixel space
    joints_new = np.zeros(yt.shape)
    joints_new[:,:,0] = yt[:,:,0]
    joints_new[:,:,1] = yt[:,:,1]
    # joints_new[:,:,2] = joints[:,:,2]
    # np.save('/mnt0/emma/IEF/tf_src/test_data/pixel_joint.npy', joints_new)
    for n in range(N):
        # if n % 10 == 0:
        #     print 'Add hms for ', n, 'th image'
        image = images[n].reshape(num_channel, img_height, img_width)
        hms = joint_to_hm(joints_new[n,:,:2], num_joints)
        out_n = np.vstack((image, hms))
        out.append(out_n)
    out = np.array(out)
    print 'Heatmaps added for', N, 'images'
    return out

def get_bounded_correction(y, yt, num_coords, L=None):
    u = y - yt
    u_norm = np.sqrt(np.sum(u**2, axis=2)).reshape(u.shape[0], u.shape[1], 1)
    mask = np.array(u_norm == 0, dtype=int)
    u_norm += mask
    unit = u / u_norm
    correction = np.zeros(unit.shape)
    if L is not None:
        u_norm = np.minimum(L, u_norm)
    for i in range(num_coords):
        correction[:,:,i] = unit[:,:,i] * u_norm.reshape(unit.shape[:2])
    return correction

def get_batch(X, y, yt, start_idx, end_idx, num_joints):
    x_batch = X[start_idx:end_idx]
    y_batch = y[start_idx:end_idx]
    yt_batch = yt[start_idx:end_idx]
    x_batch = add_hms(x_batch, yt, num_joints) # e.g. 60 x 16 x 224 x 224
    x_batch = np.swapaxes(np.swapaxes(x_batch, 1, 2), 2, 3) # e.g. 60 x 224 x 224 x 16
    # y_batch = world2pixel(y_batch)[:,:,:2] # use 2D pixel joints
    # yt_batch = world2pixel(yt_batch)[:,:,:2]
    eps_batch = get_bounded_correction(y_batch, yt_batch, num_coords=2)
    #for i in range(x_batch.shape[0]):
    #    visualizeImgHmsEps(np.copy(x_batch[i]), yt_batch[i], eps_batch[i])
    return x_batch, eps_batch

def alexNet(x, y, dropout_prob, n_outputs, input_img_size):
    # Network Parameters
    W = {}
    b = {}

    x_img = tf.reshape(x, shape=[-1,input_img_size,input_img_size,1])

    W['c1'] = tf.Variable(tf.truncated_normal([11, 11, 1, 64], stddev=0.1))
    b['c1'] = tf.Variable(tf.constant(0.1, shape=[64]))
    conv1 = tf.nn.bias_add(tf.nn.conv2d(x_img, W['c1'], strides=[1,4,4,1], padding='SAME'), b['c1'])
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    # conv1 = (29,29,64)

    W['c2'] = tf.Variable(tf.truncated_normal([5, 5, 64, 192], stddev=0.1))
    b['c2'] = tf.Variable(tf.constant(0.1, shape=[192]))
    conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1, W['c2'], strides=[1,1,1,1], padding='SAME'), b['c2'])
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    # conv2 = (15, 15, 192)

    W['c3'] = tf.Variable(tf.truncated_normal([3, 3, 192, 384], stddev=0.1))
    b['c3'] = tf.Variable(tf.constant(0.1, shape=[384]))
    conv3 = tf.nn.bias_add(tf.nn.conv2d(conv2, W['c3'], strides=[1,1,1,1], padding='SAME'), b['c3'])
    conv3 = tf.nn.relu(conv3)
    # conv3 = (15, 15, 384)

    W['c4'] = tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.1))
    b['c4'] = tf.Variable(tf.constant(0.1, shape=[256]))
    conv4 = tf.nn.bias_add(tf.nn.conv2d(conv3, W['c4'], strides=[1,1,1,1], padding='SAME'), b['c4'])
    conv4 = tf.nn.relu(conv4)
    # conv4 = (15,15,256)

    W['c5'] = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1))
    b['c5'] = tf.Variable(tf.constant(0.1, shape=[256]))
    conv5 = tf.nn.bias_add(tf.nn.conv2d(conv4, W['c5'], strides=[1,1,1,1], padding='SAME'), b['c5'])
    conv5 = tf.nn.relu(conv5)
    conv5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    # conv5 = (8,8,256)

    # Dense Layers
    W['d1'] = tf.Variable(tf.truncated_normal([8*8*256, 4096], stddev=0.1))
    b['d1'] = tf.Variable(tf.constant(0.1, shape=[4096]))
    dense1 = tf.reshape(conv5, [-1, W['d1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, W['d1']), b['d1']))
    dense1 = tf.nn.dropout(dense1, dropout_prob)

    W['d2'] = tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.1))
    b['d2'] = tf.Variable(tf.constant(0.1, shape=[4096]))
    dense2 = tf.nn.relu(tf.add(tf.matmul(dense1, W['d2']), b['d2']))
    dense2 = tf.nn.dropout(dense2, dropout_prob)

    W['out'] = tf.Variable(tf.truncated_normal([4096, n_outputs], stddev=0.1))
    b['out'] = tf.Variable(tf.constant(0.1, shape=[n_outputs]))

    y_hat = tf.add(tf.matmul(dense2, W['out']), b['out'])

    return y_hat

def vgg19(x, y, dropout_prob, n_outputs, input_img_size, num_channel):
    W = {}
    b = {}
    layer = {}

    # Make x a "square" image. Currently it's a vector
    x_img = tf.reshape(x, shape=[-1,input_img_size,input_img_size,num_channel])

    # Weight initialization
    WEIGHT_STD = 0.05

    W['c1_1'] = tf.Variable(tf.truncated_normal([3,3,16,64], stddev=WEIGHT_STD))
    b['c1_1'] = tf.Variable(tf.constant(WEIGHT_STD, shape=[64]))
    layer['c1_1'] = tf.nn.bias_add(tf.nn.conv2d(x_img, W['c1_1'], strides=[1,1,1,1], padding='SAME'), b['c1_1'])
    layer['c1_1'] = tf.nn.relu(layer['c1_1'])
    # conv1 output = (224,224,64)

    W['c1_2'] = tf.Variable(tf.truncated_normal([3,3,64,64], stddev=WEIGHT_STD))
    b['c1_2'] = tf.Variable(tf.constant(WEIGHT_STD, shape=[64]))
    layer['c1_2'] = tf.nn.bias_add(tf.nn.conv2d(layer['c1_1'], W['c1_2'], strides=[1,1,1,1], padding='SAME'), b['c1_2'])
    layer['c1_2'] = tf.nn.relu(layer['c1_2'])
    layer['c1_2'] = tf.nn.max_pool(layer['c1_2'], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # layer['c1_2'] output = (112,112,64)

    W['c2_1'] = tf.Variable(tf.truncated_normal([3,3,64,128], stddev=WEIGHT_STD))
    b['c2_1'] = tf.Variable(tf.constant(WEIGHT_STD, shape=[128]))
    layer['c2_1'] = tf.nn.bias_add(tf.nn.conv2d(layer['c1_2'], W['c2_1'], strides=[1,1,1,1], padding='SAME'), b['c2_1'])
    layer['c2_1'] = tf.nn.relu(layer['c2_1'])
    # layer['c2_1'] output = (224,224,64)

    W['c2_2'] = tf.Variable(tf.truncated_normal([3,3,128,128], stddev=WEIGHT_STD))
    b['c2_2'] = tf.Variable(tf.constant(WEIGHT_STD, shape=[128]))
    layer['c2_2'] = tf.nn.bias_add(tf.nn.conv2d(layer['c2_1'], W['c2_2'], strides=[1,1,1,1], padding='SAME'), b['c2_2'])
    layer['c2_2'] = tf.nn.relu(layer['c2_2'])
    layer['c2_2'] = tf.nn.max_pool(layer['c2_2'], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # layer['c2_2'] output = (56,56,64)

    W['c3_1'] = tf.Variable(tf.truncated_normal([3,3,128,256], stddev=WEIGHT_STD))
    b['c3_1'] = tf.Variable(tf.constant(WEIGHT_STD, shape=[256]))
    layer['c3_1'] = tf.nn.bias_add(tf.nn.conv2d(layer['c2_2'], W['c3_1'], strides=[1,1,1,1], padding='SAME'), b['c3_1'])
    layer['c3_1'] = tf.nn.relu(layer['c3_1'])

    W['c3_2'] = tf.Variable(tf.truncated_normal([3,3,256,256], stddev=WEIGHT_STD))
    b['c3_2'] = tf.Variable(tf.constant(WEIGHT_STD, shape=[256]))
    layer['c3_2'] = tf.nn.bias_add(tf.nn.conv2d(layer['c3_1'], W['c3_2'], strides=[1,1,1,1], padding='SAME'), b['c3_2'])
    layer['c3_2'] = tf.nn.relu(layer['c3_2'])

    W['c3_3'] = tf.Variable(tf.truncated_normal([3,3,256,256], stddev=WEIGHT_STD))
    b['c3_3'] = tf.Variable(tf.constant(WEIGHT_STD, shape=[256]))
    layer['c3_3'] = tf.nn.bias_add(tf.nn.conv2d(layer['c3_2'], W['c3_3'], strides=[1,1,1,1], padding='SAME'), b['c3_3'])
    layer['c3_3'] = tf.nn.relu(layer['c3_3'])

    W['c3_4'] = tf.Variable(tf.truncated_normal([3,3,256,256], stddev=WEIGHT_STD))
    b['c3_4'] = tf.Variable(tf.constant(WEIGHT_STD, shape=[256]))
    layer['c3_4'] = tf.nn.bias_add(tf.nn.conv2d(layer['c3_3'], W['c3_4'], strides=[1,1,1,1], padding='SAME'), b['c3_4'])
    layer['c3_4'] = tf.nn.relu(layer['c3_4'])
    layer['c3_4'] = tf.nn.max_pool(layer['c3_4'], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # layer['c3_4'] = (28,28,256)

    W['c4_1'] = tf.Variable(tf.truncated_normal([3,3,256,512], stddev=WEIGHT_STD))
    b['c4_1'] = tf.Variable(tf.constant(WEIGHT_STD, shape=[512]))
    layer['c4_1'] = tf.nn.bias_add(tf.nn.conv2d(layer['c3_4'], W['c4_1'], strides=[1,1,1,1], padding='SAME'), b['c4_1'])
    layer['c4_1'] = tf.nn.relu(layer['c4_1'])

    W['c4_2'] = tf.Variable(tf.truncated_normal([3,3,512,512], stddev=WEIGHT_STD))
    b['c4_2'] = tf.Variable(tf.constant(WEIGHT_STD, shape=[512]))
    layer['c4_2'] = tf.nn.bias_add(tf.nn.conv2d(layer['c4_1'], W['c4_2'], strides=[1,1,1,1], padding='SAME'), b['c4_2'])
    layer['c4_2'] = tf.nn.relu(layer['c4_2'])

    W['c4_3'] = tf.Variable(tf.truncated_normal([3,3,512,512], stddev=WEIGHT_STD))
    b['c4_3'] = tf.Variable(tf.constant(WEIGHT_STD, shape=[512]))
    layer['c4_3'] = tf.nn.bias_add(tf.nn.conv2d(layer['c4_2'], W['c4_3'], strides=[1,1,1,1], padding='SAME'), b['c4_3'])
    layer['c4_3'] = tf.nn.relu(layer['c4_3'])

    W['c4_4'] = tf.Variable(tf.truncated_normal([3,3,512,512], stddev=WEIGHT_STD))
    b['c4_4'] = tf.Variable(tf.constant(WEIGHT_STD, shape=[512]))
    layer['c4_4'] = tf.nn.bias_add(tf.nn.conv2d(layer['c4_3'], W['c4_4'], strides=[1,1,1,1], padding='SAME'), b['c4_4'])
    layer['c4_4'] = tf.nn.relu(layer['c4_4'])
    layer['c4_4'] = tf.nn.max_pool(layer['c4_4'], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # layer['c4_4'] = (14,14,512)

    W['c5_1'] = tf.Variable(tf.truncated_normal([3,3,512,512], stddev=WEIGHT_STD))
    b['c5_1'] = tf.Variable(tf.constant(WEIGHT_STD, shape=[512]))
    layer['c5_1'] = tf.nn.bias_add(tf.nn.conv2d(layer['c4_4'], W['c5_1'], strides=[1,1,1,1], padding='SAME'), b['c5_1'])
    layer['c5_1'] = tf.nn.relu(layer['c5_1'])

    W['c5_2'] = tf.Variable(tf.truncated_normal([3,3,512,512], stddev=WEIGHT_STD))
    b['c5_2'] = tf.Variable(tf.constant(WEIGHT_STD, shape=[512]))
    layer['c5_2'] = tf.nn.bias_add(tf.nn.conv2d(layer['c5_1'], W['c5_2'], strides=[1,1,1,1], padding='SAME'), b['c5_2'])
    layer['c5_2'] = tf.nn.relu(layer['c5_2'])

    W['c5_3'] = tf.Variable(tf.truncated_normal([3,3,512,512], stddev=WEIGHT_STD))
    b['c5_3'] = tf.Variable(tf.constant(WEIGHT_STD, shape=[512]))
    layer['c5_3'] = tf.nn.bias_add(tf.nn.conv2d(layer['c5_2'], W['c5_3'], strides=[1,1,1,1], padding='SAME'), b['c5_3'])
    layer['c5_3'] = tf.nn.relu(layer['c5_3'])

    W['c5_4'] = tf.Variable(tf.truncated_normal([3,3,512,512], stddev=WEIGHT_STD))
    b['c5_4'] = tf.Variable(tf.constant(WEIGHT_STD, shape=[512]))
    layer['c5_4'] = tf.nn.bias_add(tf.nn.conv2d(layer['c5_3'], W['c5_4'], strides=[1,1,1,1], padding='SAME'), b['c5_4'])
    layer['c5_4'] = tf.nn.relu(layer['c5_4'])
    layer['c5_4'] = tf.nn.max_pool(layer['c5_4'], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # layer['c5_4'] = (7,7,512)

    W['d1'] = tf.Variable(tf.truncated_normal([7*7*512, 4096], stddev=WEIGHT_STD))
    b['d1'] = tf.Variable(tf.constant(WEIGHT_STD, shape=[4096]))
    layer['d1'] = tf.reshape(layer['c5_4'], [-1, W['d1'].get_shape().as_list()[0]])
    layer['d1'] = tf.nn.relu(tf.nn.bias_add(tf.matmul(layer['d1'], W['d1']), b['d1']))
    layer['d1'] = tf.nn.dropout(layer['d1'], dropout_prob)

    W['d2'] = tf.Variable(tf.truncated_normal([4096, 4096], stddev=WEIGHT_STD))
    b['d2'] = tf.Variable(tf.constant(WEIGHT_STD, shape=[4096]))
    layer['d2'] = tf.nn.relu(tf.nn.bias_add(tf.matmul(layer['d1'], W['d2']), b['d2']))
    layer['d2'] = tf.nn.dropout(layer['d2'], dropout_prob)

    W['out'] = tf.Variable(tf.truncated_normal([4096, n_outputs], stddev=WEIGHT_STD))
    b['out'] = tf.Variable(tf.constant(WEIGHT_STD, shape=[n_outputs]))
    #layer['out'] = tf.reshape(layer['c5_4'], [-1, W['out'].get_shape().as_list()[0]])
    y_hat = tf.nn.bias_add(tf.matmul(layer['d1'], W['out']), b['out'])

    return y_hat


def main_0():
    data_root = '/mnt0/data/ITOP/out/'
    out_dir = '/mnt0/emma/IEF/tf_data/'
    views = ['top', 'side']
    val_list = range(4)
    train_list = [d + 4 for d in range(8)]

    for view in views:
        print 'View', view
        save_data(data_root, out_dir, view, val_list, 'val')
        save_data(data_root, out_dir, view, train_list, 'train')

def main_1():
    data_dir = '/mnt0/emma/IEF/tf_data/'
    img_height = 224
    img_width = 224

    resize(data_dir, img_height, img_width)

def main():
    data_root = '/mnt0/emma/IEF/tf_data/'
    image_path = data_root + 'small_train_X.npy'
    joint_path = data_root + 'small_train_y.npy'

    images = np.load(image_path)
    joints = np.load(joint_path)
    img_hms = add_hms(images, joints, num_joints=15)

    np.save('/mnt0/emma/IEF/tf_src/test_data/img_hms.npy', img_hms)

if __name__ == "__main__":
  main()
