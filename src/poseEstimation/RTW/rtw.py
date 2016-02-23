import numpy as np
import pickle
import sys
import os
from helper import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from multiprocessing import Process, Queue

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

def getInfo(depthDir, outDir, maxN, loadData=False):
	if loadData:
		I = np.load(outDir+'/data/I.npy')
		joints = np.load(outDir+'/data/joints.npy')
		theta = np.load(outDir+'/data/theta.npy')
		bodyCenters = np.load(outDir+'/data/bodyCenters.npy')
		N, _, _ = I.shape
	else:
		if not os.path.exists(outDir+'/data'):
			os.makedirs(outDir+'/data')

		# the N x H x W depth images and the N x nJoints x 3 joint locations
		I, joints = getImgsAndJoints(depthDir, maxN)
		N, _, _ = I.shape
		theta = np.random.randint(-maxOffFeat, maxOffFeat+1, (4, nFeats))
		bodyCenters = joints[:, nJoints]

		np.save(outDir+'/data/I', I)
		np.save(outDir+'/data/joints', joints[:, :nJoints])
		np.save(outDir+'/data/theta', theta)
		np.save(outDir+'/data/bodyCenters', bodyCenters)

	print '#samples: %d' % N
	return (I, joints[:, :nJoints], theta, bodyCenters, N)

'''
	The function creates the training samples. 
	Each sample is (i, q, u, f), where i is the index of the depth image, q is
	the random offset point, u is the unit direction vector toward the joint 
	location, and f is the feature array.
'''
def getSamples(outDir, jointID, theta, I, bodyCenters, joints, loadData=False):
	S_u, S_f = None, None
	nTrain, _, _ = I.shape

	if loadData:
		S_u = np.load(outDir+'/data/su'+str(jointID)+'.npy')
		S_f = np.load(outDir+'/data/sf'+str(jointID)+'.npy')
	else: 
		if not os.path.exists(outDir+'/data'):
			os.makedirs(outDir+'/data')

		S_u = np.empty((nTrain, nSamps, 3), dtype=np.float16)
		S_f = np.empty((nTrain, nSamps, nFeats), dtype=np.float16)

		for i in range(nTrain):
			if i%100 == 0:
				print 'joint %s: processing image %d/%d' % (jointName[jointID], \
																										i, nTrain)
			for j in range(nSamps):
				offsetXY = np.random.randint(-maxOffSampXY, maxOffSampXY+1, 2)
				offsetZ = np.random.uniform(-maxOffSampZ, maxOffSampZ, 1)
				offset = np.concatenate((offsetXY, offsetZ))

				S_u[i, j] = 0 if np.linalg.norm(offset) == 0 else \
											-offset/np.linalg.norm(offset)
				S_f[i, j] = getFeatures(I[i], theta, joints[i]+offset, \
											bodyCenters[i][2])

		np.save(outDir+'/data/su'+str(jointID), S_u)
		np.save(outDir+'/data/sf'+str(jointID), S_f)

	return (S_u, S_f)

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
		print 'loading model %s from files...' % jointName[jointID]
		regressor = pickle.load(open(regressorPath, 'rb'))
		L = pickle.load(open(LPath, 'rb')) 
	else:
		print 'start training model %s...' % jointName[jointID]
		regressor = DecisionTreeRegressor(min_samples_leaf=minSamplesLeaf)

		X_reshape = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
		y_reshape = y.reshape(y.shape[0]*y.shape[1], y.shape[2])

		rows = np.logical_not(np.all(X_reshape == 0, axis=1))
		regressor.fit(X_reshape[rows], y_reshape[rows])
		print 'model %s - valid samples: %d/%d' % (jointName[jointID], \
			X_reshape[rows].shape[0], X_reshape.shape[0])

		leafIDs = regressor.apply(X_reshape[rows])
		bin = np.bincount(leafIDs)
		uniqueIDs = np.unique(leafIDs)
		biggest = np.argmax(bin)
		smallest = np.argmin(bin[bin != 0])

		print 'model %s - #leaves: %d' % (jointName[jointID], uniqueIDs.shape[0])
		print 'model %s - biggest leaf id: %d, #samples: %d/%d' % \
						(jointName[jointID], biggest, bin[biggest], np.sum(bin))
		print 'model %s - smallest leaf id: %d, #samples: %d/%d' % \
						(jointName[jointID], smallest, bin[bin != 0][smallest], np.sum(bin))
		print 'model %s - average leaf size: %d' % (jointName[jointID], \
						np.sum(bin)/uniqueIDs.shape[0])

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

def trainParallel(outDir, jointID, theta, I, bodyCenters, joints, \
									loadData, loadModels, regressorQ, LQ):
	S_u, S_f = getSamples(outDir, jointID, theta, I, bodyCenters, \
												joints, loadData)

	regressor, L = trainModel(S_f, S_u, jointID, outDir, loadModels)
	regressorQ.put(regressor)
	LQ.put(L)

def main(argv):
	loadData, loadModels = False, False
	maxN = None

	depthDir = argv[0] + '/*/joints_depthcoor/*'
	outDir = argv[1]

	for i, arg in enumerate(argv[2:]):
		if arg == '-loaddata':
			loadData = True
		elif arg == '-loadmodels':
			loadModels = True
		elif arg == '-maxn':
			maxN = int(argv[2:][i+1])
			print 'maxN: %d' % maxN

	I, joints, theta, bodyCenters, N = getInfo(depthDir, outDir, maxN, loadData)

	nTrain = int(trainRatio*N)
	nTest = N - nTrain
	print 'nTrain: %d, nTest: %d' % (nTrain, nTest)

	qms = np.zeros((nTest, nJoints, nSteps+1, 3))
	joints_pred = np.zeros((nTest, nJoints, 3))

	print '\n------- training models in parallel -------'
	processes = []
	regressorQ, LQ = Queue(), Queue()
	regressors, Ls = {}, {}

	for i in range(nJoints):
		p = Process(target=trainParallel, name='Thread #%d' % i, \
								args=(outDir, i, theta, I[:nTrain], bodyCenters[:nTrain], \
								joints[:nTrain, i], loadData, loadModels, regressorQ, LQ))
		processes.append(p)
		p.start()
		regressors[i] = regressorQ.get()
		Ls[i] = LQ.get()

	[t.join() for t in processes]

	print '\n------- testing models -------'
	for idx, jointID in enumerate(kinemOrder):
		print 'testing model %s' % jointName[jointID]
		for i in range(nTest):
			qm0 = bodyCenters[nTrain:][i] if kinemParent[idx] == -1 \
							else joints_pred[i][kinemParent[idx]]
			qms[i][jointID], joints_pred[i][jointID] = testModel(
				regressors[jointID], Ls[jointID], theta, qm0, I[nTrain:][i], \
				bodyCenters[nTrain:][i])

	print 'test accuracy: %f' % getAccuracy(joints[nTrain:], joints_pred)

	# visualize predicted labels
	for i in range(nTest):
		pngPath = outDir+'/png/'+str(i)+'.png'
		if not os.path.exists(outDir+'/png'):
			os.makedirs(outDir+'/png')
		drawPred(I[nTrain:][i], joints_pred[i], qms[i], bodyCenters[nTrain:][i], \
						 pngPath)

if __name__ == '__main__':
	main(sys.argv[1:])



