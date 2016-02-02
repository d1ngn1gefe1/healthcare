import numpy as np
import process_data as pd

out_file = '/Users/Emma/GitHub/healthcare/src/jointDetector/matrix.npy'

numData = 10
d1 = 320
d2 = 240
d3 = 2
numJoints = 14
pixelPerJoint = 140

# data = np.zeros((numData,d1,d2,d3))
# data[:,:,:,0] = np.random.rand(d1,d2) * 1000
# data[:,:,:,1] = np.random.randint(numJoints+1, size=(d1,d2))

data = np.load(out_file)

(image,X,label) = pd.processData(data, numJoints, pixelPerJoint)
print image.shape
print X.shape
print label.shape

# for i in range(numJoints):
# 	num_joint = np.nonzero(label == i)[0].shape[0]
# 	print(str(i) + 'th joint: ' + str(num_joint))

# trainRatio = 0.7

# num_pixels = X.shape[0]
# num_pixel_train = round(num_pixels * trainRatio)

# xTrain = X[:num_pixel_train]
# xTest = X[num_pixel_train:]

# labelsTrain = label[:num_pixel_train]
# labelsTest = label[num_pixel_train:]

# for i in range(numJoints):
# 	num_joint = np.nonzero(labelsTrain == i)[0].shape[0]
# 	print(str(i) + 'th joint Train: ' + str(num_joint))

# for i in range(numJoints):
# 	num_joint = np.nonzero(labelsTest == i)[0].shape[0]
# 	print(str(i) + 'th joint Test: ' + str(num_joint))

