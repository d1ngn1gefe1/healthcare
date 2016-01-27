import numpy as np
import process_data as pd
from scipy.stats import multivariate_normal

N = 3 # number of epochs
T = 4 # number of iterations
K = 4 # number of joints
L = 100 # bound, maximum displacement for each joint location
M = 2 # number of images
D = 2 # dimensions of joints
W = 224 # width of input image
H = 224 # height of input image

cov = [[1, 0], [0, 1]] # covariance matrix for Gaussian heatmaps

y0 = np.zeros((K, 2)) # initial mean pose
I = np.zeros((M, 3, H, W)) # training images 

'''
Input:
- joints: A numpy array of shape K x D
Output: 
- heatmaps: A numpy array of shape K x H x W
'''
def joints2Heatmaps(joints):
	pair = np.nonzero(np.ones((H, W)))
	idx = np.array(zip(pair[0], pair[1])).reshape(H, W, D)
	heatmaps = []

	for k in range(K):
	    mean = joints[k]
	    heatmap = multivariate_normal.pdf(idx, mean, cov)
	    heatmaps.append(heatmap)

	heatmaps = np.array(heatmaps)
 	return heatmaps

'''
Input:
- y: ground truth joint locations (M x K x D)
- yt: predicted joint locations in the t-th iteration (M x K x D)
Output: 
- epsilon: M x K x D
'''
def getTargetBoundedCorrections(y, yt): # e(y, yt)
	u = y - yt # M x K x D
	uNorm = np.linalg.norm(u, axis=2).clip(max=L) # M x K
	uUnit = (u.reshape(M*K, D)/(uNorm.reshape(M*K)[:, np.newaxis])).reshape(M, K, D)
	epsilon = (uNorm.reshape(M*K)[:, np.newaxis]*uUnit.reshape(M*K, D)).reshape(M, K, D)

	return epsilon

'''
Input:
- I: training images (M x 3 x H x W)
- yt: predicted joint locations in the t-th iteration (M x K x D)
Output:
- Xt: M x (K+3) x H x W
'''	
def yt2Xt(I, yt): # g()
	Xt = np.zeros((M, K+3, H, W))
	Xt[:, :3] = I

	for m in range(M):
		Xt[m, 3:K+3] = joints2Heatmaps(yt[m]) # K x H x W
		if m % 100 == 99:
			print '\t%d00th image' % ((m+1)/100)

	return Xt

#####

yt = y0
I = np.random.rand(M, 3, H, W)
yt = np.random.rand(M, K, D)
y = np.random.rand(M, K, D)
for t in range(1):
	print '%dth iteration' % (t+1)
	#Xt = yt2Xt(I, yt)
	epsilon = getTargetBoundedCorrections(y, yt)
	print yt
	print '\n\n\n'
	print y
	print '\n\n\n'
	print epsilon

	''' ConvNet: 
	Input: 
	- X: A numpy array of shape M x (K+3) x H x W containing the training data. 
	     Each image is a concatenation of the RGB image I and the Gaussian 
	     heatmaps g.
	- y: A numpy array of shape M x K x D containing the training labels. Each
	     label is the corrections of joints locations in the corresponding 
	     image.
	Output:
	- epsilon: A numpy array of shape M x K x D containing the predicted 
	           corrections.
	'''
	#for n in range(N):
	#	ConvNet.train(X, y)
	#epsilon = ConvNet.test(X)
	#yt = yt + epsilon
	#X = yt2Xt(I, yt)



