import numpy as np
from sklearn.ensemble import RandomForestClassifier
from util import load_data, processData
from get_acc_joints import *
import os

# parameters
numJoints = 14
pixelPerJoint = 140
trainRatio = 0.7
root_dir = '/mnt0/emma/shotton/data_ext/'
data_dirs = ['data1/']
data_dirs = [root_dir + d for d in data_dirs]
out_path = '/mnt0/emma/shotton/src/out/'
BATCH_SIZE = 500
train_batch = 1
test_batch = 1

def get_data(batch, train_flag, root_dir=root_dir, data_dir=data_dirs):
  data = {}
  features = []
  labels = []
  for directory in data_dir:
    dirs = os.listdir(directory)
    dirs = [directory+d+'/' for d in dirs if d.find('0') != -1]
    dirs.sort()
    if train_flag:
      del dirs[-1] # reserve the last one for testing
      if batch > len(dirs):
        batch -= len(dirs)
        for d in dirs:
          labels.append(np.load(d + 'labels.npy'))
	  features.append(np.load(d + 'features.npy'))
          print d, 'train loaded!'
      else:
        for i in range(batch):
	  labels.append(np.load(dirs[i] + 'labels.npy'))
	  features.append(np.load(dirs[i] + 'features.npy'))
	  print i, 'train loaded!'
        break
    else:
      if batch > 0:
        labels.append(np.load(dirs[-1] + 'labels.npy'))
        features.append(np.load(dirs[-1] + 'features.npy'))
        print batch, 'test loaded!'
        batch -= 1
      else:
        break
  # features = np.vstack(features)
  length = [len(f) for f in features]
  feature_stack = np.zeros((sum(length), 2000))
  label_stack = np.zeros((sum(length),))
  start = 0
  for i in range(len(length)):
    feature_stack[start:start+length[i]:] = features[i]
    label_stack[start:start+length[i]:] = labels[i]
    start += length[i]
  data['features'], data['labels'] = feature_stack, label_stack
  return data


# No. training images: batch * BATCH_SIZE
train_data = get_data(batch=train_batch, train_flag=True)
'''
train_data = {}
train_data['features'] = np.zeros((605431+609567, 2000))
train_data['labels'] = np.zeros((605431+609567,))
train_data['features'][:605431,:] = np.load('/mnt0/emma/shotton/data_ext/data1/00/features.npy')
train_data['labels'][:605431,] = np.load('/mnt0/emma/shotton/data_ext/data1/00/labels.npy')
train_data['features'][605431:,:] = np.load('/mnt0/emma/shotton/data_ext/data1/01/features.npy')
train_data['labels'][605431:,] = np.load('/mnt0/emma/shotton/data_ext/data1/01/labels.npy')
'''
num_train_images = train_batch * BATCH_SIZE
print 'Num training images:', num_train_images 
  
for i in range(numJoints):
  num = np.nonzero(train_data['labels'] == i)[0].shape[0]
  print('Joint ' + str(i) + ': ' + str(num))

# train a random forest classifier
print 'Feature shape:', train_data['features'].shape
print 'Label shape:', train_data['labels'].shape
rf = RandomForestClassifier(n_estimators=3, criterion='entropy', max_depth=20)
rf.fit(train_data['features'], train_data['labels'])

del train_data

# prediction
test_data = get_data(batch=test_batch, train_flag=False)
pred_label = rf.predict(test_data['features'])
pred_prob = rf.predict_proba(test_data['features'])
pred_label_path = out_path + 'pred_label_'+str(num_train_images)+'.npy'
pred_prob_path = out_path + 'pred_prob_'+str(num_train_images)+'.npy'
np.save(pred_label_path, pred_label)
np.save(pred_prob_path, pred_prob)

# accuracy
score = rf.score(test_data['features'], test_data['labels'])
print('Score: ' + str(score))

avg_accuracy = util.get_cm_acc(test_data['labels'].astype(int), pred_label)
print('Avg accuracy: ' + str(avg_accuracy))

accuracy = 0.0
accClass = np.zeros((numJoints,2))
for i in range(0,pred_label.shape[0]):
  accClass[int(test_data['labels'][i])][1] += 1
  if pred_label[i] == test_data['labels'][i]:
    accClass[pred_label[i]][0] += 1
    accuracy += 1
accuracy /= float(pred_label.shape[0])

for i in range(0,numJoints):
  print('Class ' + str(i) + ': ' + str(accClass[i][0]) + '/' + str(accClass[i][1]))
print('Total accuracy: ' + str(accuracy))  
