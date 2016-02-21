import numpy as np
import helper
import sys
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
np.set_printoptions(threshold=np.nan)

nSamps = 1000 # the number of samples of each joint
nFeats = 1000 # the number of features of each offset point
maxOffSampXY = 25 # the maximum offset for samples in x, y axes
maxOffSampZ = 2 # the maximum offset for samples in z axis
maxOffFeat = 100 # the maximum offset for features (before divided by d)
largeNum = 100
nSteps = 128
stepSize = 1
rmZeros = True # remove features with all zeros
H = 240
W = 320 
K = 10
minSamplesLeaf = 100

'''
	The function creates the training samples. 
	Each sample is (i, q, u, f), where i is the index of the depth image, q is
	the random offset point, u is the unit direction vector toward the joint 
	location, and f is the feature array.
'''
def getSamples(dataDir, featsDir, maxN, load=False):
	S_i, S_q, S_u, S_f = None, None, None, None

	if load:
		I = np.load(featsDir+'/I.npy')
		theta = np.load(featsDir+'/theta.npy')
		bodyCenters = np.load(featsDir+'/bodyCenters.npy')
		S_i = np.load(featsDir+'/si.npy')
		S_q = np.load(featsDir+'/sq.npy')
		S_u = np.load(featsDir+'/su.npy')
		S_f = np.load(featsDir+'/sf.npy')

		N, _, _ = I.shape
		nJoints, _ = S_i.shape
	else: 
		# the N x H x W depth images and the N x nJoints x 3 joint locations
		I, joints = helper.getImgsAndJoints(dataDir, maxN)
		N, _, _ = I.shape
		_, nJoints, _ = joints.shape
		nJoints -= 1

		# t1x, t1y, t2x, t2y
		theta = np.random.randint(-maxOffFeat, maxOffFeat+1, (4, nFeats))
		
		S_i = np.empty((nJoints, N*nSamps), dtype=np.int16)
		S_q = np.empty((nJoints, N*nSamps, 3), dtype=np.int16)
		S_u = np.empty((nJoints, N*nSamps, 3), dtype=np.float16)
		S_f = np.empty((nJoints, N*nSamps, nFeats), dtype=np.float16)

		bodyCenters = joints[:, nJoints]

		for i in range(nJoints):
			for j in range(N):
				if j%100 == 0:
					print 'processing joint %d, image %d' % (i+1, j+1)
				for k in range(nSamps):
					offsetXY = np.random.randint(-maxOffSampXY, maxOffSampXY+1, 2)
					offsetZ = np.random.uniform(-maxOffSampZ, maxOffSampZ, 1)
					offset = np.concatenate((offsetXY, offsetZ))

					S_i[i][j*nSamps+k] = j
					S_q[i][j*nSamps+k] = joints[j, i] + offset
					S_u[i][j*nSamps+k] = 0 if np.linalg.norm(offset) == 0 else \
																-offset/np.linalg.norm(offset)
					S_f[i][j*nSamps+k] = getFeatures(I[i], theta, j, joints[j, i]+offset)

		np.save(featsDir+'/I', I)
		np.save(featsDir+'/theta', theta)
		np.save(featsDir+'/bodyCenters', bodyCenters)
		np.save(featsDir+'/si', S_i)
		np.save(featsDir+'/sq', S_q)
		np.save(featsDir+'/su', S_u)
		np.save(featsDir+'/sf', S_f)

	print 'Dimensions: N=%d, H=%d, W=%d, nJoints=%d' % (N, H, W, nJoints)
	return (I, theta, bodyCenters, S_i, S_q, S_u, S_f, N, nJoints)

def getFeatures(d, theta, i, q):
	d[d == 0] = largeNum
	coor = q[:2][::-1] # coor: y, x
	coor[0] = np.clip(coor[0], 0, H-1)
	coor[1] = np.clip(coor[1], 0, W-1)
	dq = d[tuple(coor)]

	x1 = np.clip(coor[1]+theta[0]/dq, 0, W-1).astype(int)
	y1 = np.clip(coor[0]+theta[1]/dq, 0, H-1).astype(int)
	x2 = np.clip(coor[1]+theta[2]/dq, 0, W-1).astype(int)
	y2 = np.clip(coor[0]+theta[3]/dq, 0, H-1).astype(int)

	return d[y1, x1] - d[y2, x2]

def stochastic(regressor, features, unitDirections, K):
	indices = regressor.apply(features) # leaf id of each sample
	leafIDs = np.unique(indices) # array of unique leaf ids
	L = {}

	for leafID in leafIDs:
		kmeans = KMeans(n_clusters=K)
		labels = kmeans.fit_predict(unitDirections[indices == leafID])
		weights = np.bincount(labels).astype(float)/labels.shape[0]
		centers = kmeans.cluster_centers_
		centers /= np.linalg.norm(centers, axis=1)[:, np.newaxis]
		#helper.checkUnitVectors(centers)

		L[leafID] = (weights, centers)

	return L

def getUnitDirection(l):
	K = l[0].shape[0]
	idx = np.random.choice(K, p=l[0])
	return l[1][idx]

def main(argv):
	load, train = False, False

	dataDir = argv[0] + '/*/joints_depthcoor/*'
	featsDir = argv[1]

	maxN = None
	for i, arg in enumerate(argv[2:]):
		if arg == '-load':
			load = True
		elif arg == '-train':
			train = True
		elif arg == '-N':
			maxN = int(argv[2:][i+1])
			print 'maxN: %d' % maxN

	I, theta, bodyCenters, S_i, S_q, S_u, S_f, N, nJoints = \
		getSamples(dataDir, featsDir, maxN, load)
	print 'N: %d, nJoints: %d' % (N, nJoints)

	if not train:
		return

	'''
	S_i_train = S_i[:, :nTrain]
	S_q_train = S_q[:, :nTrain]
	S_u_train = S_u[:, :nTrain]
	S_f_train = S_f[:, :nTrain]
	S_i_test = S_i[:, nTrain:]
	S_q_test = S_q[:, nTrain:]
	S_u_test = S_u[:, nTrain:]
	S_f_test = S_f[:, nTrain:]
	'''

	kinemOrder =   [0, 12, 13, 1, 2, 5, 3, 6, 4, 7,  8, 10, 9, 11]
	kinemParent = [-1, -1, -1, 0, 0, 0, 2, 5, 3, 6, 12, 13, 8, 10]

	for i in range(93,94):
		qm = np.empty((nJoints, nSteps+1, 3))
		jointsPred = np.empty((nJoints, 3))
		for idx, j in enumerate(kinemOrder):
			print '\n\n'
			features, unitDirections = None, None
			if rmZeros:
				rows = np.logical_not(np.all(S_f[j] == 0, axis=1))
				features = S_f[j][rows]
				unitDirections = S_u[j][rows]
			else:
				features = S_f[j]
				unitDirections = S_u[j]
			
			regressor = DecisionTreeRegressor(min_samples_leaf=minSamplesLeaf)
			regressor.fit(features, unitDirections)

			leafIDs = regressor.apply(features)
			bin = np.bincount(leafIDs)
			uni = np.unique(leafIDs)
			biggest = np.argmax(bin)
			smallest = np.argmin(bin[bin != 0])
			#print bin
			#print uni
			print '#leaves: %d' % uni.shape
			print 'biggest leaf id: %d, #samples: %d/%d' % \
				(biggest, bin[biggest], N*nSamps)
			print 'smallest leaf id: %d, #samples: %d/%d' % \
				(smallest, bin[bin != 0][smallest], N*nSamps)
			print 'average leaf size: %d' % (N*nSamps/uni.shape[0])
			L = stochastic(regressor, features, unitDirections, K)
			print 'length of dict: %d' % len(L)

			if kinemParent[idx] == -1:
				qm[j][0] = bodyCenters[i]
			else:
				qm[j][0] = jointsPred[kinemParent[idx]]
			qsum = np.zeros(3)

			for k in range(nSteps):
				f = getFeatures(I[i], theta, j, qm[j][k]).reshape(1, -1)
				leafID = regressor.apply(f)[0]
				u = getUnitDirection(L[leafID])
				qm[j][k+1] = qm[j][k] + u*stepSize
				qm[j][k+1][0] = np.clip(qm[j][k+1][0], 0, W-1)
				qm[j][k+1][1] = np.clip(qm[j][k+1][1], 0, H-1)
				qsum += qm[j][k+1]

			jointsPred[j] = qsum/nSteps
			#print jointsPred[j]
			#helper.drawPts(I[i], jointsPred[j])
			#helper.drawPts(I[i], qm[j])

		helper.drawPred(I[i], jointsPred, qm, bodyCenters[i], 
										featsDir+'/'+str(i)+'.png')

	#helper.drawPts(I[idx], jointsPred)

if __name__ == "__main__":
	main(sys.argv[1:])



