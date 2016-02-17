import numpy as np
import helper
from sklearn.tree import DecisionTreeRegressor

nOffPts = 100 # the number of offset points of each joint
nFeats = 500 # the number of features of each offset point
maxOffSampXY = 30 # the maximum offset for samples in x, y axes
maxOffSampZ = 5 # the maximum offset for samples in z axis
maxOffFeat = 30 # the maximum offset for features
epsilon = 1e-8 # prevent divide-by-zero
largeNum = 1e3
maxDepth = 13
nSteps = 128
stepSize = 2
featsDir = '/Users/alan/Documents/research/seq_c_1/'

# the N x H x W depth images and the N x nJoints x 3 joint locations
I, joints = helper.getImgsAndJoints()
N, H, W = I.shape
_, nJoints, _ = joints.shape
print 'Dimensions: %d, %d, %d, %d' % (N, H, W, nJoints)

minSamplesLeaf = max(2, N*nOffPts/2**maxDepth)
nTrain = int(N*nOffPts*0.9)
# t1x, t1y, t2x, t2y
theta = np.random.randint(-maxOffFeat, maxOffFeat+1, (4, nFeats))

'''
	The function creates the training samples. 
	Each sample is (i, q, u, f), where i is the index of the depth image, q is
	the random offset point, u is the unit direction vector toward the joint 
	location, and f is the feature array.
'''
def getSamples(load=False):
	S_i, S_q, S_u, S_f = None, None, None, None
	if load:
		S_i = np.load(featsDir+'si.npy')
		S_q = np.load(featsDir+'sq.npy')
		S_u = np.load(featsDir+'su.npy')
		S_f = np.load(featsDir+'sf.npy')
	else: 
		S_i = np.empty((nJoints, N*nOffPts)).astype(int)
		S_q = np.empty((nJoints, N*nOffPts, 3)).astype(int)
		S_u = np.empty((nJoints, N*nOffPts, 3)).astype(float)
		S_f = np.empty((nJoints, N*nOffPts, nFeats)).astype(float)

		for i in range(nJoints):
			for j in range(N):
				for k in range(nOffPts):
					offsetXY = np.random.randint(-maxOffSampXY, maxOffSampXY+1, 2)
					offsetZ = np.random.uniform(-maxOffSampZ, maxOffSampZ, 1)
					offset = np.concatenate((offsetXY, offsetZ))
					S_i[i][j*nOffPts+k] = j
					S_q[i][j*nOffPts+k] = joints[j, i] + offset
					S_u[i][j*nOffPts+k] = offset/(np.linalg.norm(offset)+epsilon)
					S_f[i][j*nOffPts+k] = getFeatures(j, joints[j, i]+offset)

		np.save(featsDir+'si', S_i)
		np.save(featsDir+'sq', S_q)
		np.save(featsDir+'su', S_u)
		np.save(featsDir+'sf', S_f)

	return (S_i, S_q, S_u, S_f)

def getFeatures(i, q):
	d = I[i]
	coor = q[:2][::-1].astype(int)

	dx = largeNum if d[tuple(coor)] == 0 else d[tuple(coor)]
	x1 = np.clip(coor[0]+theta[0]/dx, 0, W-1).astype(int)
	y1 = np.clip(coor[1]+theta[1]/dx, 0, H-1).astype(int)
	x2 = np.clip(coor[0]+theta[2]/dx, 0, W-1).astype(int)
	y2 = np.clip(coor[1]+theta[3]/dx, 0, H-1).astype(int)

	return d[y1, x1] - d[y2, x2]

S_i, S_q, S_u, S_f = getSamples(True)
S_i_train = S_i[:, :nTrain]
S_q_train = S_q[:, :nTrain]
S_u_train = S_u[:, :nTrain]
S_f_train = S_f[:, :nTrain]
S_i_test = S_i[:, nTrain:]
S_q_test = S_q[:, nTrain:]
S_u_test = S_u[:, nTrain:]
S_f_test = S_f[:, nTrain:]


for i in range(1):
	regressor = DecisionTreeRegressor(max_depth=maxDepth, \
																		min_samples_leaf=minSamplesLeaf)
	regressor.fit(S_f_train[i], S_u_train[i])

	qm = np.empty((nSteps+1, 3)).astype(float)
	qm[0] = [180, 95, -5]
	qsum = np.zeros(3)
	for j in range(nSteps):
		f = getFeatures(10, qm[j]).reshape(1, -1)
		u = regressor.predict(f).ravel()
		print u
		qm[j+1] = qm[j] + u*stepSize
		qsum += qm[j+1]
	q = qsum/nSteps
	print q
	helper.drawPts(I[10], qm)





