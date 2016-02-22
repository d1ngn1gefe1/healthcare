import glob
import numpy as np
import cv2

H = 240
W = 320 
nJoints = 14

palette = [(34, 88, 226), (34, 69, 101), (0, 195, 243), (146, 86, 135), \
					 (0, 132, 243), (241, 202, 161), (50, 0, 190), (128, 178, 194), \
					 (23, 45, 136), (86, 136, 0), (172, 143, 230), (165, 103, 0), \
					 (121, 147, 249), (151, 78, 96), (0, 166, 246), (108, 68, 179), \
					 (0, 211, 220), (130, 132, 132), (0, 182, 141), (38, 61, 43)] # BGR
jointName = ["NECK", "HEAD", "LEFT SHOULDER", "LEFT ELBOW", \
						 "LEFT HAND", "RIGHT SHOULDER", "RIGHT ELBOW", "RIGHT HAND", \
						 "LEFT KNEE", "LEFT FOOT", "RIGHT KNEE", "RIGHT FOOT", \
						 "LEFT HIP", "RIGHT HIP", "TORSO"]
kinemOrder =   [0, 1, 2, 5, 3, 6, 4, 7, 12, 13,  8, 10, 9, 11]
kinemParent = [-1, 0, 0, 0, 2, 5, 3, 6, -1, -1, 12, 13, 8, 10]

def getImgsAndJoints(dataDir, maxN, noBg=True):	
	jointsPaths = glob.glob(dataDir)
	total = len(jointsPaths) 

	I = np.empty((total, H, W)).astype('float16')
	joints = np.empty((total, nJoints+1, 3))

	idx = 0
	for i in range(total):
		if i%100 == 0:
			print 'loading image %d' % (i+1)
		tmp = np.loadtxt(jointsPaths[i])
		if tmp.shape[0] == nJoints-2:
			imgPath = jointsPaths[i].replace('txt', 'npy') \
								 .replace('joints_depthcoor', 'nparray_depthcoor')
			I[idx] = np.load(imgPath).astype('float16')
			joints[idx, 0:nJoints-2, :] = tmp
			idx += 1
			if idx == maxN:
				break

	joints[:, nJoints-2, :] = (joints[:, 0, :]+2*joints[:, 8, :])/3
	joints[:, nJoints-1, :] = (joints[:, 0, :]+2*joints[:, 10, :])/3
	joints[:, nJoints, :] = (2*joints[:, 0, :]+joints[:, nJoints-2, :]+
														 joints[:, nJoints-1, :])/4

	print 'total number of images: %d/%d' % (idx, total)
	I = I[:idx]
	joints = joints[:idx]
	if noBg:
		I = bgSub(I, joints)
	return (I, joints) # including bodyCenters

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
			cv2.circle(img, tuple(joint[:2].astype(np.uint16)), 2, (0,0,255), -1)
		cv2.imshow('image', img)
		cv2.waitKey(0)
	cv2.destroyAllWindows()

def drawPts(img, pts):
	img = cv2.equalizeHist(img.astype(np.uint8))
	img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
	if pts.ndim == 1:
		cv2.circle(img, tuple(pts[:2].astype(np.uint16)), 4, (255,0,0), -1)
	else:
		nPts = pts.shape[0]
		for i, pt in enumerate(pts):
			color = (255*(nPts-i)/nPts, 0, 255*i/nPts)
			cv2.circle(img, tuple(pt[:2].astype(np.uint16)), 1, color, -1)
	cv2.imshow('image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def drawPred(img, joints, paths, center, filename):
	H = img.shape[0]
	W = img.shape[1]

	#img = cv2.equalizeHist(img.astype(np.uint8))
	img = img.astype(np.uint8)
	img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
	img = np.hstack((img, np.zeros((H, 100, 3)))).astype(np.uint8)

	for i, path in enumerate(paths):
		nPts = path.shape[0]
		for j, pt in enumerate(path):
			color = tuple(c*(2*j+nPts)/(3*nPts) for c in palette[i]) 
			cv2.circle(img, tuple(pt[:2].astype(np.uint16)), 1, color, -1)

	for i, joint in enumerate(joints):
		cv2.circle(img, tuple(joint[:2].astype(np.uint16)), 4, palette[i], -1)

	cv2.rectangle(img, 
								tuple([int(center[0]-2), int(center[1]-2)]), 
								tuple([int(center[0]+2), int(center[1]+2)]), 
								palette[nJoints], -1)
	#cv2.imshow('image', img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	for i, joint in enumerate(joints):
		cv2.rectangle(img, (W, H*i/nJoints), (W+100, H*(i+1)/nJoints-1), 
									palette[i], -1)
		cv2.putText(img, jointName[i], (W, H*(i+1)/nJoints-5), 
								cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))

	cv2.imwrite(filename, img)

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
