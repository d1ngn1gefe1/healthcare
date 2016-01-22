import numpy as np
from os import listdir
from sklearn.metrics import confusion_matrix

data_dir = '../data/'
out_file = './out/matrix.npy'
width = 320
height = 240

def load_data():
  depth = []
  label = []

  for doc in listdir(data_dir):
    if (doc.find('depth') != -1):
      depth.append(np.loadtxt(data_dir + doc, delimiter='\n').reshape(width, height))
    elif (doc.find('label') != -1):
      label.append(np.loadtxt(data_dir + doc, delimiter='\n').reshape(width, height))

  depth = np.array(depth)
  label = np.array(label)
  data = np.empty((depth.shape[0], width, height, 2))
  data[:, :, :, 0] = depth
  data[:, :, :, 1] = label

  np.save(out_file, data)
  print('Data saved!')
  return data

# data is 1068 x 320 x 240 x 2, numJoint = 14
def processData(data, numJoint, pixelPerJoint):
  numImage = data.shape[0]
  image = data[:,:,:,0]
  labelAll = data[:,:,:,1]

  X = []
  label = []
  depth = []
  index_image = []

  for i in range(numImage):
    # each joint should have roughly the same number of pixels
    jointNumPixel = np.zeros((numJoint))
    for j in range(numJoint):
      jointNumPixel[j] = np.nonzero(labelAll[i] == j)[0].shape[0] # joint label: 0,...,13

    jointNumPixel = np.minimum(pixelPerJoint, jointNumPixel)
    for j in range(numJoint):
      pair = np.nonzero(labelAll[i] == j)
      indices = np.column_stack((pair[0], pair[1]))
      depth_val = image[i][indices[:,0], indices[:,1]]

      X += indices[0:jointNumPixel[j], :].tolist()
      depth += depth_val[0:jointNumPixel[j]].tolist()
      index_image += (i * np.ones((jointNumPixel[j]))).tolist()
      label += np.extract(labelAll[i] == j, labelAll[i])[0:jointNumPixel[j]].tolist()

  X = np.array(X)
  depth = np.array(depth)
  depth = depth.reshape(depth.shape[0], 1)
  index_image = np.array(index_image)
  index_image = index_image.reshape(index_image.shape[0], 1)
  np.save('index_image.npy', index_image)

  X = np.append(np.append(X, depth, axis=1), index_image, axis=1) # each row of X: (x,y,depth,#image)
  label = np.array(label)
  np.save('X.npy', X)
  return (image, X, label)

def part_to_joint(X, label, prob, num_data, num_joints):
  joints = np.zeros((num_data, num_joints, 3))

  for i in range(num_data): # for each image in the test set
    indices = np.nonzero(X[:,3] == i)[0]
    if (indices.size != 0):
      for j in range(num_joints):
        indices_joint = indices[np.nonzero(label[indices] == j)[0]]
        if (indices_joint.size == 0):
	  joints[i, j] = -1 # if the i-th image doesn't have pixels labeled as the jth joint, the joint coords will be (-1,-1,-1)
        else:
          joint_coords = X[indices_joint][:,:3]
	  print('Joint coords shape: ' + str(joint_coords.shape))
          joint_probs = prob[indices_joint, j].reshape(indices_joint.size, 1)
      	  print('Joint prob shape: ' + str(joint_probs.shape))
	  print(str(j) + 'th joint prob: ' + str(joint_probs))
          normalize = np.sum(joint_probs)
	  print('Normalize: ' + str(normalize))
	  if (normalize > 0):
            joints[i, j] = np.sum(joint_coords * joint_probs) / normalize
	  else:
	    joints[i, j] = -1
  return joints

def partition_data(data, pixel_per_joint, num_joints, train_ratio):
  (images, X, labels) = processData(data, num_joints, pixel_per_joint)
  num_pixels = X.shape[0]
  num_pixel_train = round(num_pixels * train_ratio)

  labels_train = labels[:num_pixel_train]
  labels_test = labels[num_pixel_train:]

  X_train = X[:num_pixel_train]
  X_test = X[num_pixel_train:]

  return (X_train, X_test, labels_train, labels_test)


def get_cm_acc(y_true, y_pred):
  cm = confusion_matrix(y_true, y_pred)
  cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  num_classes = cm_normalized.shape[0]
  avg_accuracy = np.trace(cm_normalized) / num_classes
  return avg_accuracy

def get_joints(data, pred_prob, num_joints, pixel_per_joint, train_ratio):
  (X_train, X_test, labels_train, labels_test) = partition_data(data, pixel_per_joint, num_joints, train_ratio)
  num_data = data.shape[0]
  joints = part_to_joint(X_test, labels_test, pred_prob, num_data, num_joints)
  return joints


