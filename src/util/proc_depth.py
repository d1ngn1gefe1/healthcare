import numpy as np
import glob
import os
import cv2

W = 320
H = 240
inDirs = glob.glob('/mnt0/data/EVAL/data')

C = 3.8605e-3 #NUI_CAMERA_DEPTH_NOMINAL_INVERSE_FOCAL_LENGTH_IN_PIXELS


for inDir in inDirs:
	inDir = inDir + '/depth/'
	outDir = inDir + '/depth_frames/' 

	if not os.path.exists(outDir):
    	os.makedirs(outDir)

	paths = glob.glob(inDir + '*.txt')
	for i, path in enumerate(paths):
		fName = path.replace(inDir, '').replace('txt', 'jpg')
		if i%100 == 0:
			print 'image %d: %s' % (i+1, fName)

		f = np.loadtxt(path)

		indices = np.nonzero(f[:, 2])
		worldX = f[:, 0][indices]
		worldY = f[:, 1][indices]
		worldZ = f[:, 2][indices]
		#print np.amin(worldX), np.amax(worldX)
		#print np.amin(worldY), np.amax(worldY)
		#print np.amin(worldZ), np.amax(worldZ)

		depthZ = worldZ
		depthX = worldX/worldZ/C + W/2.0
		depthY = worldY/worldZ/C + H/2.0
		#print np.amin(depthX), np.amax(depthX)
		#print np.amin(depthY), np.amax(depthY)
		#print np.amin(depthZ), np.amax(depthZ)

		depthX = np.clip(np.rint(depthX), 0, W-1).astype(np.int32)
		depthY = np.clip(np.rint(depthY), 0, H-1).astype(np.int32)
		zMin = np.amin(depthZ)
		zMax = np.amax(depthZ)
		depthZ = (depthZ-zMin)/(zMax-zMin)*255

		#print np.amin(depthX), np.amax(depthX)
		#print np.amin(depthY), np.amax(depthY)
		#print np.amin(depthZ), np.amax(depthZ)

		out = np.zeros((H, W))
		out[depthY, depthX] = depthZ

		out = cv2.equalizeHist(out.astype(np.uint8))
		cv2.imwrite(outDir + fName, out)
		#cv2.imshow('image', out)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
