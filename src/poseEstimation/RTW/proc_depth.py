import numpy as np
import glob
import os
import cv2
import sys

W = 320
H = 240
#datasets = glob.glob('/mnt0/data/EVAL/data/*')

C = 3.8605e-3 #NUI_CAMERA_DEPTH_NOMINAL_INVERSE_FOCAL_LENGTH_IN_PIXELS

def main(argv):
	getJpg, getJoints, getNpArray = False, False, False

	datasets = glob.glob(argv[0])
	print datasets

	for arg in argv[1:]:
		if arg == '-jpg':
			getJpg = True
		elif arg == '-joints':
			getJoints = True
		elif arg == '-nparray':
			getNpArray = True

	for dataset in datasets:
		inImgsDir = dataset + '/depth/'
		inJointsDir = dataset + '/joints/'
		outJpgDir = dataset + '/jpg_depthcoor/'
		outNpArrayDir = dataset + '/nparray_depthcoor/' 
		outJointsDir = dataset + '/joints_depthcoor/' 

		if getJoints:
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

		if getJpg or getNpArray:
			if not os.path.exists(outJpgDir) and getJpg:
				os.makedirs(outJpgDir)
			if not os.path.exists(outNpArrayDir) and getNpArray:
				os.makedirs(outNpArrayDir)

			paths = glob.glob(inImgsDir + '*.txt')
			for i, path in enumerate(paths):
				fName = path.replace(inImgsDir, '').replace('.txt', '')
				if i%100 == 0:
					print 'image %d: %s' % (i+1, fName)

				if getJpg:
					outJpg = np.zeros((H, W))
				if getNpArray:
					outNpArray = np.zeros((H, W))

				f = np.loadtxt(path)

				indices = np.nonzero(f[:, 2])

				if len(indices[0]) != 0:
					worldX = f[:, 0][indices]
					worldY = f[:, 1][indices]
					worldZ = f[:, 2][indices]

					depth = worldZ
					print worldX/worldZ
					depthX = worldX/worldZ/C + W/2.0
					depthY = worldY/worldZ/C + H/2.0

					depthX = np.clip(np.rint(depthX), 0, W-1).astype(np.int32)
					depthY = np.clip(np.rint(depthY), 0, H-1).astype(np.int32)

					if getJpg:
						zMin = np.amin(depth)
						zMax = np.amax(depth)
						depth = (depth-zMin)/(zMax-zMin)*255
						outJpg[depthY, depthX] = depth
						outJpg = cv2.equalizeHist(outJpg.astype(np.uint8))
					if getNpArray:
						outNpArray[depthY, depthX] = worldZ

				if getJpg:
					#cv2.imwrite(outJpgDir+fName+'.jpg', outJpg)
					cv2.imshow('img', outJpg)
					cv2.waitKey(0)
				if getNpArray:	
					np.save(outNpArrayDir+fName, outNpArray)

if __name__ == "__main__":
	main(sys.argv[1:])