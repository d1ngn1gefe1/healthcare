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

data_dir = '../data/'
out_image_path = data_dir + 'images.npy'
true_joint_path = data_dir + 'joints.npy'
out_joint_path = 'pred_joint.npy'
num_joints = 14
T = 4 # number of iterations
img_width = 224
img_height = 224
train_ratio = 0.8
BATCH_SIZE_TEST = 50
eps = 1e-5

images = np.load(out_image_path)
joints = np.load(true_joint_path)
num_images = images.shape[0]
num_train = round(num_images * train_ratio)

test_images = images[num_train:]
true_joints = joints[num_train:]
num_test = test_images.shape[0]
y0 = np.ones((num_images, num_joints, 2)) # initial mean pose
y0 = y0 * (true_joints[0]+eps) # initial mean pose
batches = int(num_test / BATCH_SIZE_TEST)

print 'Test image shape:', test_images.shape

y_t = y0
for t in range(T):
  test_data = yt2Xt(test_images, y_t, img_height, img_width,
                    num_test, num_joints, 2) # 400 x 17 x 224 x 224
  npy_to_h5(test_data, y_t, train_flag=False)
  print t, ': data saved!'

  # load ief model
  solver = caffe.SGDSolver(caffe_root + solvertxt)
  solver.net.copy_from(caffe_root + iefnet)

  predicted = []
  for b in xrange(batches):
    data4D = np.zeros([BATCH_SIZE_TEST,17,224,224])
    data4DL = np.zeros([BATCH_SIZE_TEST,28])
    data4D[0:BATCH_SIZE_TEST,:] = test_data[b*BATCH_SIZE_TEST:b*BATCH_SIZE_TEST+BATCH_SIZE_TEST,:]
    npy_to_h5(data4D, data4DL, train_flag=False)
    print 'batch ', b, data4D.shape, data4DL.shape

    #predict
    solver.test_nets[0].forward()
    pred = solver.test_nets[0].blobs['loss3/regression'].data
    print 'pred[0]: ', pred[0]
    predicted.append(pred)

  predicted = np.asarray(predicted, 'float32')
  predicted = predicted.reshape(batches*BATCH_SIZE_TEST,28)
  print 'Total in Batches ', data4D.shape, batches
  print 'Predicted shape: ', predicted.shape

  y_t += predicted.reshape(y_t.shape)

print 'Final joint prediction shape: ', y_t.shape
np.save(data_dir + out_joint_path, y_t)
