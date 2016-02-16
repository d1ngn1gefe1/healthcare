import numpy as np
import glob
import os
import cv2

W = 320
H = 240
datasets = glob.glob('/mnt0/data/EVAL/data/*')
#print datasets

C = 3.8605e-3 #NUI_CAMERA_DEPTH_NOMINAL_INVERSE_FOCAL_LENGTH_IN_PIXELS

for dataset in datasets:
	inImgsDir = dataset + '/depth/'
	inJointsDir = dataset + '/joints/'
	outImgsDir = dataset + '/imgs_depthcoor/'
	outJointsDir = dataset + '/joints_depthcoor/' 

	if not os.path.exists(outImgsDir):
		os.makedirs(outImgsDir)
	if not os.path.exists(outJointsDir):
		os.makedirs(outJointsDir)

	paths = glob.glob(inJointsDir + '*.txt')
	for i, path in enumerate(paths):
		fName = path.replace(inJointsDir, '')
		if i%100 == 0:
			print 'joint %d: %s' % (i+1, fName)

		f = np.loadtxt(path)
		if f.shape[0] == 0:
			continue

		worldX = f[:, 0]
		worldY = f[:, 1]
		worldZ = f[:, 2]

		depthZ = worldZ
		depthX = worldX/worldZ/C + W/2.0
		depthY = worldY/worldZ/C + H/2.0

		depthX = np.clip(np.rint(depthX), 0, W-1).astype(np.int32)
		depthY = np.clip(np.rint(depthY), 0, H-1).astype(np.int32)

		out = np.column_stack((depthX, depthY, depthZ))
		np.savetxt(outJointsDir + fName, out)

'''
	paths = glob.glob(inImgsDir + '*.txt')
	for i, path in enumerate(paths):
		fName = path.replace(inImgsDir, '').replace('txt', 'jpg')
		if i%100 == 0:
			print 'image %d: %s' % (i+1, fName)

		out = np.zeros((H, W))
		f = np.loadtxt(path)

		indices = np.nonzero(f[:, 2])

		if len(indices[0]) == 0:
			cv2.imwrite(outImgsDir + fName, out)
			continue

		worldX = f[:, 0][indices]
		worldY = f[:, 1][indices]
		worldZ = f[:, 2][indices]
		#print np.amin(worldX), np.amax(worldX)
		#print np.amin(worldY), np.amax(worldY)
		#print np.amin(worldZ), np.amax(worldZ)

		depth = worldZ
		depthX = worldX/worldZ/C + W/2.0
		depthY = worldY/worldZ/C + H/2.0
		#print np.amin(depthX), np.amax(depthX)
		#print np.amin(depthY), np.amax(depthY)
		#print np.amin(depth), np.amax(depth)

		depthX = np.clip(np.rint(depthX), 0, W-1).astype(np.int32)
		depthY = np.clip(np.rint(depthY), 0, H-1).astype(np.int32)
		zMin = np.amin(depth)
		zMax = np.amax(depth)
		depth = (depth-zMin)/(zMax-zMin)*255

		#print np.amin(depthX), np.amax(depthX)
		#print np.amin(depthY), np.amax(depthY)
		#print np.amin(depth), np.amax(depth)

		out[depthY, depthX] = depth

		out = cv2.equalizeHist(out.astype(np.uint8))
		cv2.imwrite(outImgsDir + fName, out)
		#cv2.imshow('image', out)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
'''
