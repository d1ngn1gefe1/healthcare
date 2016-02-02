import numpy as np
import util
from os import listdir
from map_features import map_features_thread

num_joints = 14
pixel_per_joint = 140
train_ratio = 0.7
data_saved = True
out_path = './out/'
feature_path = 'features/'
maxOffset = 100 # max offset for u and v
num_features = 2000

# phi = (theta, tau)
theta_u = np.random.randint(maxOffset, size=(num_features,2)) # each row is (u1,u2)
theta_v = np.random.randint(maxOffset, size=(num_features,2)) # each row is (v1,v2)

if data_saved:
  data = np.load(out_path + 'matrix.npy')
else:
  data = util.load_data()

features = listdir(feature_path)
num_threads = len(features)
if num_threads > 0:
  print('here!')
  feature_all = []
  for i in range(num_threads): 
    feature = np.load(feature_path + str(i) + '.npy')
    feature_all.append(feature)
  feature_all = np.array(feature_all)
  feature_all = np.vstack(feature_all)
  np.save(out_path + 'feature_all_1000.npy', feature_all)
else:
  (images, X, labels) = util.processData(data, num_joints, pixel_per_joint)
  map_features_thread(X, theta_u, theta_v, images)
