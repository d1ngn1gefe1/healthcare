import numpy as np
import sys
from process_data import *
from ief import *

# Caffe Settings
sys.path.append('/opt/caffe/python/caffe')
caffe_root = '/opt/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

niter = 150 # num_train = 1600, batch_size = 32, epoch = 3
# losses will also be stored in the log
train_loss1 = np.zeros(niter)
train_loss2 = np.zeros(niter)
train_loss3 = np.zeros(niter)
solver = caffe.SGDSolver(caffe_root + 'models/finetune_ggnet_ief/solver.prototxt')
solver.net.copy_from(caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel')

data_dir = '../data/'
image_path = data_dir + 'lsp_images/'
joint_path = data_dir + 'joints.txt'
out_image_path = data_dir + 'images.npy'
out_joint_path = data_dir + 'joints.npy'

N = 3 # number of epochs
T = 1 # number of iterations
num_joints = 14
img_width = 224
img_height = 224
train_ratio = 0.8

images = np.load(out_image_path) # 2000 x 3 x 224 x 224
num_images = images.shape[0]
y0 = np.zeros((num_image, num_joints, 2)) # initial mean pose
y = np.load(out_joint_path) # true joint location: 2000 x 14 x 2
num_train = round(num_images * train_ratio)
num_test = num_images - num_train

train_images = images[:num_train]
train_y = y[:num_train]
train_y0 = y0[:num_train]
test_images = images[num_train:]
test_y = y[num_train:]
test_y0 = y0[num_train:]

y_t = train_y0
for t in range(T):
  epsilon_t = getTargetBoundedCorrections(train_y, y_t, num_train, num_joints, 2) # 1600 x 14 x 2
  train_data = yt2Xt(train_images, y_t, img_height, img_width) # 1600 x 17 x 224 x 224
  npy_to_h5(train_data, epsilon_t, train_flag=True) # save data and label to h5
  for it in range(niter):
    solver.step(1)  # SGD by Caffe
    # store the train loss
    train_loss1[it] = solver.net.blobs['loss1/loss1'].data
    train_loss2[it] = solver.net.blobs['loss2/loss1'].data
    train_loss3[it] = solver.net.blobs['loss3/loss3'].data
    if it % 10 == 0:
      print('iter' + str(it) +
            ', regression_loss=' +
            str(train_loss1[it]) + ','+
            str(train_loss2[it]) + ',' +
            str(train_loss3[it]))
  print 'done'
  solver.net.forward()
  epsilon_pred = solver.net.blobs['loss3/regression'].data
  yt += epsilon_pred

solver.net.save(caffe_root + 'models/finetune_ggnet_ief/ief.caffemodel')
