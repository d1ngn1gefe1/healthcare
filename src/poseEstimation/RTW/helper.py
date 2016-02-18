import glob
import numpy as np
import cv2

dataDir = '/mnt0/data/EVAL/data'
H = 240
W = 320 
nJoints = 12

def getImgsAndJoints():	
	imgsPaths = glob.glob(dataDir + '/jpg_depthcoor/*')
	jointsPaths = glob.glob(dataDir + '/joints_depthcoor/*')
	N = len(imgsPaths)

	I = np.empty((N, H, W))
	joints = np.empty((N, nJoints, 3))

	idx = 0
	for i in range(N):
		tmp = np.loadtxt(jointsPaths[i])
		if tmp.shape[0] == nJoints:
			I[idx] = np.load(imgsPaths[i])
			joints[idx] = tmp
			idx += 1

	return (I[:idx], joints[:idx])

def visualizeImgs(I, joints):
	N = I.shape[0]

	for i in range(N):
		img = I[i]
		img = cv2.equalizeHist(img.astype(np.uint8))
		for joint in joints[i]:
			cv2.circle(img, tuple(joint[:2].astype(np.uint8)), 2, 255, -1)
		cv2.imshow('image', img)
		cv2.waitKey(0)
	cv2.destroyAllWindows()

def drawPts(img, pts):
	img = cv2.equalizeHist(img.astype(np.uint8))
	img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
	nPts = pts.shape[0]
	for i, pt in enumerate(pts):
		cv2.circle(img, tuple(pt[:2].astype(np.uint8)), 1, (0,0,255*i/nPts), -1)
	cv2.imshow('image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#testing
'''
I, joints = getImgsAndJoints()
print I.shape, joints.shape
visualizeImgs(I, joints)
'''
