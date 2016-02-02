import numpy as np
import sys
from process_data import *
from ief import *

# Caffe Settings
sys.path.append('/opt/caffe/python/caffe')
caffe_root = '/opt/caffe/'
googlenet = 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
iefnet = 'models/finetune_ggnet_ief/iefnet.caffemodel'
solvertxt = 'models/finetune_ggnet_ief/solver.prototxt'
sys.path.insert(0, caffe_root + 'python')
import caffe

niter = 150 # num_train = 1600, batch_size = 32, epoch = 3
# losses will also be stored in the log
train_loss1 = np.zeros(niter)
train_loss2 = np.zeros(niter)
train_loss3 = np.zeros(niter)

data_dir = '../data/'
image_path = data_dir + 'lsp_images/'
joint_path = data_dir + 'joints.txt'
out_image_path = data_dir + 'images.npy'
out_joint_path = data_dir + 'joints.npy'

N = 3 # number of epochs
T = 4 # number of iterations
num_joints = 14
img_width = 224
img_height = 224
train_ratio = 0.8
BATCH_SIZE_TRAIN = 32
eps = 1e-5

images = np.load(out_image_path) # 2000 x 3 x 224 x 224
num_images = images.shape[0]
y = np.load(out_joint_path) # true joint location: 2000 x 14 x 2
y0 = np.ones((num_images, num_joints, 2)) # initial mean pose
y0 = y0 * (y[0]+eps) # initial mean pose
num_train = round(num_images * train_ratio)
num_test = num_images - num_train

train_images = images[:num_train]
train_y = y[:num_train]
train_y0 = y0[:num_train]
test_images = images[num_train:]
test_y = y[num_train:]
test_y0 = y0[num_train:]

epsilon_test_0 = getTargetBoundedCorrections(test_y, test_y0, num_test, num_joints, 2)
test_data = yt2Xt(test_images, test_y0, img_height, img_width, num_test, num_joints, 2)
npy_to_h5(test_data, epsilon_test_0, train_flag=False)

y_t = train_y0
batches = int(num_train / BATCH_SIZE_TRAIN)

for t in range(T):
  epsilon_t = getTargetBoundedCorrections(train_y, y_t, num_train, num_joints, 2) # 1600 x 14 x 2
  train_data = yt2Xt(train_images, y_t, img_height, img_width, 
                     num_train, num_joints, 2) # 1600 x 17 x 224 x 224
  npy_to_h5(train_data, epsilon_t, train_flag=True) # save data and label to h5
  print(str(t) +': data saved!')

  if t == 0:
    model = googlenet
  else:
    model = iefnet
  solver = caffe.SGDSolver(caffe_root + solvertxt)
  solver.net.copy_from(caffe_root + model)
  
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
  solver.net.save(caffe_root + iefnet)

  predicted = []
  for b in xrange(batches): 
    data4D = np.zeros([BATCH_SIZE_TRAIN,17,224,224])
    data4DL = np.zeros([BATCH_SIZE_TRAIN,28])
    data4D[0:BATCH_SIZE_TRAIN,:] = train_data[b*BATCH_SIZE_TRAIN:b*BATCH_SIZE_TRAIN+BATCH_SIZE_TRAIN,:]
    npy_to_h5(data4D, data4DL, train_flag=True)
    print 'batch ', b, data4D.shape, data4DL.shape
    solver = caffe.SGDSolver(caffe_root + solvertxt)
    solver.net.copy_from(caffe_root + model)
    
    #predict
    pred = solver.net.forward()

    predicted.append(solver.net.blobs['loss3/regression'].data)

  predicted = np.asarray(predicted, 'float32')
  predicted = predicted.reshape(batches*BATCH_SIZE_TRAIN,28)

  print 'Total in Batches ', data4D.shape, batches
  print 'Predicted shape: ', predicted.shape
  
  y_t += predicted.reshape(y_t.shape)
