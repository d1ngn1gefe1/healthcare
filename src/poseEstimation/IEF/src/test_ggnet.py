import sys
import numpy as np

# Caffe Settings
sys.path.append('/opt/caffe/python/caffe')
caffe_root = '/opt/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()
# load the model
net = caffe.Net(caffe_root + 'models/bvlc_googlenet/deploy.prototxt',
                caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel',
		caffe.TEST)

# load input and configure preprocessing
# Informing the transformer of the necessary input shape
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# Set the mean to normalize the data
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
# Defining the order of the channels in input data (doesn't matter for grayscale images) 
transformer.set_transpose('data', (2,0,1))
# The reference model has channels in BGR order instead of RGB
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

#note we can change the batch size on-the-fly
#since we classify only one image, we change batch size from 10 to 1
net.blobs['data'].reshape(1,3,224,224)

#load the image in the data layer
im = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
net.blobs['data'].data[...] = transformer.preprocess('data', im)

#compute
out = net.forward()

#predicted predicted class
print out['prob'].argmax()

#print predicted labels
labels = np.loadtxt(caffe_root + "data/ilsvrc12/synset_words.txt", str, delimiter='\t')
top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
print labels[top_k]

