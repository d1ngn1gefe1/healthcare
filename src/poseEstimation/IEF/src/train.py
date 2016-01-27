import numpy as np
import sys

# Caffe Settings
sys.path.append('/opt/caffe/python/caffe')
caffe_root = '/opt/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

niter = 200
# losses will also be stored in the log
train_loss1 = np.zeros(niter)
train_loss2 = np.zeros(niter)
train_loss3 = np.zeros(niter)

solver = caffe.SGDSolver(caffe_root + 'models/finetune_ggnet_ief/solver.prototxt')
solver.net.copy_from(caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel')

for it in range(niter):
    solver.step(1)  # SGD by Caffe
    # store the train loss
    train_loss1[it] = solver.net.blobs['loss1/loss1'].data
    train_loss2[it] = solver.net.blobs['loss2/loss1'].data
    train_loss3[it] = solver.net.blobs['loss3/loss3'].data

    if it % 10 == 0:
        print('iter' + str(it) + 
              ', finetune_loss=' + 
              str(train_loss1[it]) + ','+ 
              str(train_loss2[it]) + ',' + 
              str(train_loss3[it]))
print 'done'
