import numpy as np
import sys
from process_data import *

# Caffe Settings
sys.path.append('/opt/caffe/python/caffe')
caffe_root = '/opt/caffe/'
googlenet = 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
iefnet = 'models/finetune_ggnet_ief/iefnet_50.caffemodel'
solvertxt = 'models/finetune_ggnet_ief/solver.prototxt'
sys.path.insert(0, caffe_root + 'python')
import caffe

niter = 20 # num_train = 8000, batch_size = 32, epoch = 3
# losses will also be stored in the log
train_loss1 = np.zeros(niter)
train_loss2 = np.zeros(niter)
train_loss3 = np.zeros(niter)

data_dir = '../data/lsp_ext/'
image_path = data_dir + 'lsp_images/'
out_image_path = data_dir + 'images.npy'
out_joint_path = data_dir + 'joints.npy'

N = 3 # number of epochs
T = 1 # number of iterations
L = 20 # maximum displacement for key-point joints
num_joints = 14
img_width = 224
img_height = 224
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 50

images = np.load(out_image_path)[:2000] # 2000 x 3 x 224 x 224
num_images = images.shape[0]
y = np.load(out_joint_path)[:2000] # true joint location: 2000 x 14 x 2
y0 = np.ones(y.shape)
num_train = 64
num_test = 50

train_images = images[:num_train]
train_y = y[:num_train]
y_median = np.median(train_y, axis=0)
train_y0 = y0[:num_train] * y_median # initialize mean pose
test_images = images[num_train:num_train+num_test]
test_y = y[num_train:num_train+num_test]
test_y0 = y0[num_train:num_train+num_test] * y_median

# image preprocessing
mean_image = np.mean(train_images, axis=0)
train_images = train_images - mean_image
test_images = test_images - mean_image
train_images /= 255
test_images /= 255

y_t = train_y0
test_yt = test_y0
batches = int(num_train / BATCH_SIZE_TRAIN) # 50
batches_test = int(num_test / BATCH_SIZE_TEST) # 8

for t in range(T):
  epsilon_t = get_bounded_correction(train_y, y_t, L) # 1600 x 14 x 2
  # normalize label to be in [-1,1]
  epsilon_t /= L 
  np.save(data_dir + 'eps_'+str(t)+'.npy', epsilon_t)
  train_data = add_hms(train_images, y_t, num_joints) # 1600 x 17 x 224 x 224
  npy_to_h5(train_data, epsilon_t, train_flag=True) # save data and label to h5
  print(str(t) +': train data saved!')

  # validation data:
  eps_test = get_bounded_correction(test_y, test_yt, L)
  eps_test /= L
  test_data = add_hms(test_images, test_yt, num_joints)
  npy_to_h5(test_data, eps_test, train_flag=False)
  print(str(t) +': test data saved!')

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
    if it % 2 == 0:
      print('t' + str(t) + ', iter' + str(it) +
            ', regression_loss=' +
            str(train_loss1[it]) + ','+
            str(train_loss2[it]) + ',' +
            str(train_loss3[it]))
  # print 'done'
  solver.net.save(caffe_root + iefnet)

  predicted = []
  predicted_test = []
  for b in xrange(batches): 
    data4D = np.zeros([BATCH_SIZE_TRAIN,17,224,224])
    data4DL = np.zeros([BATCH_SIZE_TRAIN,28])
    data4D[0:BATCH_SIZE_TRAIN,:] = train_data[b*BATCH_SIZE_TRAIN:b*BATCH_SIZE_TRAIN+BATCH_SIZE_TRAIN,:]
    npy_to_h5(data4D, data4DL, train_flag=True)
    if b < batches_test:
      data4D = np.zeros([BATCH_SIZE_TEST,17,224,224])
      data4DL = np.zeros([BATCH_SIZE_TEST,28])
      data4D[0:BATCH_SIZE_TEST,:] = test_data[b*BATCH_SIZE_TEST:b*BATCH_SIZE_TEST+BATCH_SIZE_TEST,:]
      npy_to_h5(data4D, data4DL, train_flag=False)
    # print 'batch ', b, data4D.shape, data4DL.shape
    solver = caffe.SGDSolver(caffe_root + solvertxt)
    solver.net.copy_from(caffe_root + iefnet)
    
    #predict
    pred = solver.net.forward()
    predicted.append(solver.net.blobs['loss3/regression'].data)
    if b < batches_test:
      pred_test = solver.test_nets[0].forward()
      predicted_test.append(solver.test_nets[0].blobs['loss3/regression'].data)

  predicted = np.asarray(predicted, 'float32')
  predicted = predicted.reshape(batches*BATCH_SIZE_TRAIN, 28)
  np.save(data_dir+'pred_'+str(t)+'.npy', predicted)

  predicted_test = np.asarray(predicted_test, 'float32')
  predicted_test = predicted_test.reshape(batches_test*BATCH_SIZE_TEST, 28)
  np.save(data_dir+'pred_test_'+str(t)+'.npy', predicted_test)

  # print 'Total in Train Batches ', data4D.shape, batches
  print 'Train Predicted shape: ', predicted.shape
  print 'Test Predicted shape: ', predicted_test.shape
  
  predicted *= L
  y_t += predicted.reshape(y_t.shape)

  predicted_test *= L
  test_yt += predicted_test.reshape(test_yt.shape)
  
  count, accuracy, avg_acc = eval_accuracy(train_y, y_t)
  print 'Correct count per joint train: ', count
  print 'Accuracy per joint train: ', accuracy
  print 'Average accuracy train: ', avg_acc

  count_test, accuracy_test, avg_acc_test = eval_accuracy(test_y, test_yt)
  print 'Correct count per joint test: ', count_test
  print 'Accuracy per joint test: ', accuracy_test
  print 'Average accuracy test: ', avg_acc_test

np.save('pred_train.npy', y_t)
np.save('pred_test.npy', test_yt)


