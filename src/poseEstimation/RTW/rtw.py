import numpy as np
from helper import *
import sys
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
import pickle

np.set_printoptions(threshold=np.nan)

nSamps = 1000 # the number of samples of each joint
nFeats = 1000 # the number of features of each offset point
maxOffSampXY = 50 # the maximum offset for samples in x, y axes
maxOffSampZ = 2 # the maximum offset for samples in z axis
maxOffFeat = 200 # the maximum offset for features (before divided by d)
largeNum = 100
nSteps = 200
stepSize = 2
K = 10
minSamplesLeaf = 200
trainRatio = 0.9
tolerance = 20

'''
	The function creates the training samples. 
	Each sample is (i, q, u, f), where i is the index of the depth image, q is
	the random offset point, u is the unit direction vector toward the joint 
	location, and f is the feature array.
'''
def getSamples(dataDir, outDir, maxN, loadSamples=False):
	S_i, S_q, S_u, S_f = None, None, None, None

	if loadSamples:
		I = np.load(outDir+'/I.npy')
		joints = np.load(outDir+'/joints.npy')
		theta = np.load(outDir+'/theta.npy')
		bodyCenters = np.load(outDir+'/bodyCenters.npy')
		S_i = np.load(outDir+'/si.npy')
		S_q = np.load(outDir+'/sq.npy')
		S_u = np.load(outDir+'/su.npy')
		S_f = np.load(outDir+'/sf.npy')
		N, _, _ = I.shape
	else: 
		# the N x H x W depth images and the N x nJoints x 3 joint locations
		I, joints = getImgsAndJoints(dataDir, maxN)
		N, _, _ = I.shape

		# t1x, t1y, t2x, t2y
		theta = np.random.randint(-maxOffFeat, maxOffFeat+1, (4, nFeats))
		
		S_i = np.empty((nJoints, N, nSamps), dtype=np.int16)
		S_q = np.empty((nJoints, N, nSamps, 3), dtype=np.int16)
		S_u = np.empty((nJoints, N, nSamps, 3), dtype=np.float16)
		S_f = np.empty((nJoints, N, nSamps, nFeats), dtype=np.float16)

		bodyCenters = joints[:, nJoints]

		for jointID in range(nJoints):
			for i in range(N):
				if i%100 == 0:
					print 'processing joint %d, image %d' % (jointID+1, i+1)
				for j in range(nSamps):
					offsetXY = np.random.randint(-maxOffSampXY, maxOffSampXY+1, 2)
					offsetZ = np.random.uniform(-maxOffSampZ, maxOffSampZ, 1)
					offset = np.concatenate((offsetXY, offsetZ))

					S_i[jointID,i,j] = i
					S_q[jointID,i,j] = joints[i, jointID] + offset
					S_u[jointID,i,j] = 0 if np.linalg.norm(offset) == 0 else \
															 -offset/np.linalg.norm(offset)
					S_f[jointID,i,j] = getFeatures(I[i], theta, \
															 joints[i, jointID]+offset, \
															 bodyCenters[i][2])

		np.save(outDir+'/I', I)
		np.save(outDir+'/joints', joints[:, :nJoints])
		np.save(outDir+'/theta', theta)
		np.save(outDir+'/bodyCenters', bodyCenters)
		np.save(outDir+'/si', S_i)
		np.save(outDir+'/sq', S_q)
		np.save(outDir+'/su', S_u)
		np.save(outDir+'/sf', S_f)

	print '#samples: %d' % N
	return (I, joints[:, :nJoints], theta, bodyCenters, S_i, S_q, S_u, S_f, N)

def getFeatures(img, theta, q, z):
	img[img == 0] = largeNum
	coor = q[:2][::-1] # coor: y, x
	coor[0] = np.clip(coor[0], 0, H-1)
	coor[1] = np.clip(coor[1], 0, W-1)
	dq = z if img[tuple(coor)] == largeNum else img[tuple(coor)]

	x1 = np.clip(coor[1]+theta[0]/dq, 0, W-1).astype(int)
	y1 = np.clip(coor[0]+theta[1]/dq, 0, H-1).astype(int)
	x2 = np.clip(coor[1]+theta[2]/dq, 0, W-1).astype(int)
	y2 = np.clip(coor[0]+theta[3]/dq, 0, H-1).astype(int)

	return img[y1, x1] - img[y2, x2]

def stochastic(regressor, features, unitDirections):
	indices = regressor.apply(features) # leaf id of each sample
	leafIDs = np.unique(indices) # array of unique leaf ids
	L = {}

	for leafID in leafIDs:
		kmeans = KMeans(n_clusters=K)
		labels = kmeans.fit_predict(unitDirections[indices == leafID])
		weights = np.bincount(labels).astype(float)/labels.shape[0]
		centers = kmeans.cluster_centers_
		centers /= np.linalg.norm(centers, axis=1)[:, np.newaxis]
		#checkUnitVectors(centers)

		L[leafID] = (weights, centers)

	return L

def trainModel(X, y, jointID, outDir, loadModels=False):
	regressor, L = None, None

	if not os.path.exists(outDir+'/models'):
		os.makedirs(outDir+'/models')
	regressorPath = outDir + '/models/regressor' + str(jointID) + '.pkl'
	LPath = outDir + '/models/L' + str(jointID) + '.pkl'

	if loadModels:
		regressor = pickle.load(open(regressorPath, 'rb'))
		L = pickle.load(open(LPath, 'rb')) 
	else:
		print '\n------- joint %s -------' % jointName[jointID]
		regressor = DecisionTreeRegressor(min_samples_leaf=minSamplesLeaf)

		X_reshape = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
		y_reshape = y.reshape(y.shape[0]*y.shape[1], y.shape[2])

		rows = np.logical_not(np.all(X_reshape == 0, axis=1))
		regressor.fit(X_reshape[rows], y_reshape[rows])
		print 'valid samples: %d/%d' % (X_reshape[rows].shape[0], 
			X_reshape.shape[0])

		leafIDs = regressor.apply(X_reshape[rows])
		bin = np.bincount(leafIDs)
		uniqueIDs = np.unique(leafIDs)
		biggest = np.argmax(bin)
		smallest = np.argmin(bin[bin != 0])

		print '#leaves: %d' % uniqueIDs.shape[0]
		print 'biggest leaf id: %d, #samples: %d/%d' % \
						(biggest, bin[biggest], np.sum(bin))
		print 'smallest leaf id: %d, #samples: %d/%d' % \
						(smallest, bin[bin != 0][smallest], np.sum(bin))
		print 'average leaf size: %d' % (np.sum(bin)/uniqueIDs.shape[0])

		L = stochastic(regressor, X_reshape, y_reshape)

		pickle.dump(regressor, open(regressorPath, 'wb')) 
		pickle.dump(L, open(LPath, 'wb')) 

	return regressor, L

def testModel(regressor, L, theta, qm0, img, bodyCenter):
	qm = np.zeros((nSteps+1, 3))
	qm[0] = qm0
	joint_pred = np.zeros(3)

	for i in range(nSteps):
		f = getFeatures(img, theta, qm[i], bodyCenter[2]).reshape(1, -1)
		leafID = regressor.apply(f)[0]

		idx = np.random.choice(K, p=L[leafID][0])
		u = L[leafID][1][idx]

		qm[i+1] = qm[i] + u*stepSize
		qm[i+1][0] = np.clip(qm[i+1][0], 0, W-1)
		qm[i+1][1] = np.clip(qm[i+1][1], 0, H-1)
		joint_pred += qm[i+1]

	joint_pred = joint_pred/nSteps

	return (qm, joint_pred)

def getAccuracy(joints, joints_pred):
	joints_reshape = joints.reshape(joints.shape[0]*joints.shape[1], 
																	joints.shape[2])
	joints_pred_reshape = joints_pred.reshape(joints_pred.shape[0]\
													*joints_pred.shape[1], joints_pred.shape[2])
	assert joints_reshape.shape[0] == joints_pred_reshape.shape[0]
	dists = np.sqrt(np.sum((joints_reshape-joints_pred_reshape)**2, axis=1))
	return float(np.sum(dists < tolerance))/joints_reshape.shape[0]

def main(argv):
	loadSamples, loadModels, train = False, False, False
	maxN = None

	dataDir = argv[0] + '/*/joints_depthcoor/*'
	outDir = argv[1]

	for i, arg in enumerate(argv[2:]):
		if arg == '-loadsamples':
			loadSamples = True
		elif arg == '-loadmodels':
			loadModels = True
		elif arg == '-train':
			train = True
		elif arg == '-maxn':
			maxN = int(argv[2:][i+1])
			print 'maxN: %d' % maxN

	I, joints, theta, bodyCenters, S_i, S_q, S_u, S_f, N = \
		getSamples(dataDir, outDir, maxN, loadSamples)

	if not train:
		return

	nTrain = int(trainRatio*N)
	nTest = N - nTrain
	print 'nTrain: %d, nTest: %d' % (nTrain, nTest)

	S_i_train = S_i[:, :nTrain]
	S_q_train = S_q[:, :nTrain]
	S_u_train = S_u[:, :nTrain]
	S_f_train = S_f[:, :nTrain]
	I_train = I[:nTrain]
	joints_train = joints[:nTrain]
	bodyCenters_train = bodyCenters[:nTrain]
	S_i_test = S_i[:, nTrain:]
	S_q_test = S_q[:, nTrain:]
	S_u_test = S_u[:, nTrain:]
	S_f_test = S_f[:, nTrain:]
	I_test = I[nTrain:]
	joints_test = joints[nTrain:]
	bodyCenters_test = bodyCenters[nTrain:]

	qms = np.zeros((nTest, nJoints, nSteps+1, 3))
	joints_pred = np.zeros((nTest, nJoints, 3))
	for idx, jointID in enumerate(kinemOrder):
		regressor, L = trainModel(S_f_train[jointID], S_u_train[jointID], 
															jointID, outDir, loadModels)
		for i in range(nTest):
			qm0 = bodyCenters_test[i] if kinemParent[idx] == -1 \
							else joints_pred[i][kinemParent[idx]]
			qms[i][jointID], joints_pred[i][jointID] = testModel(regressor, L, \
																									theta, qm0, I_test[i], \
																									bodyCenters_test[i])

	print 'test accuracy: %f' % getAccuracy(joints_test, joints_pred)

	for i in range(nTest):
		pngPath = outDir+'/png/'+str(i)+'.png'
		if not os.path.exists(outDir+'/png'):
			os.makedirs(outDir+'/png')
		drawPred(I_test[i], joints_pred[i], qms[i], bodyCenters_test[i], pngPath)

if __name__ == "__main__":
	main(sys.argv[1:])



