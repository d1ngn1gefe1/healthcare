import glob
import numpy as np
import cv2

H = 240
W = 320 
nJoints = 12

def getImgsAndJoints(dataDir, N, noBg=True):	
	jointsPaths = glob.glob(dataDir)
	total = len(jointsPaths) 

	I = np.empty((total, H, W)).astype('float16')
	joints = np.empty((total, nJoints+3, 3))

	idx = 0
	for i in range(total):
		if i%100 == 0:
			print 'loading image %d' % (i+1)
		tmp = np.loadtxt(jointsPaths[i])
		if tmp.shape[0] == nJoints:
			imgPath = jointsPaths[i].replace('txt', 'npy') \
								 .replace('joints_depthcoor', 'nparray_depthcoor')
			I[idx] = np.load(imgPath).astype('float16')
			joints[idx, 0:nJoints, :] = tmp
			idx += 1
			if idx == N:
				break

	joints[:, nJoints, :] = (joints[:, 0, :]+2*joints[:, 8, :])/3
	joints[:, nJoints+1, :] = (joints[:, 0, :]+2*joints[:, 10, :])/3
	joints[:, nJoints+2, :] = (2*joints[:, 0, :]+joints[:, nJoints, :]+
														 joints[:, nJoints+1, :])/4

	print 'total number of images: %d/%d' % (idx, total)
	I = I[:idx]
	joints = joints[:idx]
	if noBg:
		I = bgSub(I, joints)
	return (I, joints)

def bgSub(I, joints):
	thres = 250
	scale = 30

	N = I.shape[0]
	assert N == joints.shape[0]
	indices = np.indices(I.shape[1:]).swapaxes(0, 1).swapaxes(1, 2)
	mask = 1e10*np.ones(I.shape)

	# scale z axis
	I = scale*I
	joints_copy = joints.copy()
	joints_copy[:, :, 2] = scale*joints_copy[:, :, 2]
	joints_copy[:, :, [0, 1]] = joints_copy[:, :, [1, 0]]

	for i in range(N):
		mat = np.concatenate((indices, I[i][:, :, np.newaxis]), 2)
		for j in range(joints_copy.shape[1]):
			mask[i] = np.minimum(mask[i], np.sum(np.square(mat-joints_copy[i][j]), 2))

	mask[mask > thres] = 0
	mask[mask > 0] = 1
	return I*mask

def visualizeImgs(I, joints):
	N = I.shape[0]

	for i in range(N):
		img = I[i]
		img = cv2.equalizeHist(img.astype(np.uint8))
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
		for joint in joints[i]:
			cv2.circle(img, tuple(joint[:2].astype(np.uint8)), 2, (0,0,255), -1)
		cv2.imshow('image', img)
		cv2.waitKey(0)
	cv2.destroyAllWindows()

def drawPts(img, pts):
	img = cv2.equalizeHist(img.astype(np.uint8))
	img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
	if pts.ndim == 1:
		cv2.circle(img, tuple(pts[:2].astype(np.uint8)), 4, (255,0,0), -1)
	else:
		nPts = pts.shape[0]
		for i, pt in enumerate(pts):
			color = (255*(nPts-i)/nPts, 0, 255*i/nPts)
			cv2.circle(img, tuple(pt[:2].astype(np.uint8)), 1, color, -1)
	cv2.imshow('image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def drawPred(img, joints, paths, center):
	#img = cv2.equalizeHist(img.astype(np.uint8))
	img = img.astype(np.uint8)
	img[img > 0] = 255
	img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
	for path in paths:
		nPts = path.shape[0]
		for i, pt in enumerate(path):
			color = (255*(nPts-i)/nPts, 0, 255*i/nPts)
			cv2.circle(img, tuple(pt[:2].astype(np.uint8)), 1, color, -1)
	for joint in joints:
		cv2.circle(img, tuple(joint[:2].astype(np.uint8)), 4, (0,255,0), -1)
	cv2.circle(img, tuple(center[:2].astype(np.uint8)), 4, (128,128,0), -1)
	cv2.imshow('image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def checkUnitVectors(unitVectors):
	s1 = np.sum(unitVectors.astype(np.float32)**2)
	s2 = unitVectors.shape[0]
	print 'error: %0.3f' % (abs(s1-s2)/s2)


#testing
'''
I, joints = getImgsAndJoints()
print I.shape, joints.shape
mask = bgSub(I, joints)
visualizeImgs(mask, joints)
'''
