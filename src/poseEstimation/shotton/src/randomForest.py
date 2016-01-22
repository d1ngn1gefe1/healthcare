import numpy as np
from sklearn.ensemble import RandomForestClassifier
from process_data import load_data, processData, part_to_joint
from map_features import map_features

# parameters
numJoints = 14
pixelPerJoint = 140
num_features = 2000 # dimension of feature vector after feature mapping
maxOffset = 100 # max offset for u and v
trainRatio = 0.7
out_file = './out/matrix.npy'
feature_train_path = './out/feature_train.npy'
feature_test_path = './out/feature_test.npy'

saved = True
mapped = False

# load data
if not saved:
    data = load_data()
else:
    data = np.load(out_file)
    print('Data loaded!')

# data = data[:100]
numData = data.shape[0]
print('Num data: ' + str(numData))

(images, X, labels) = processData(data, numJoints, pixelPerJoint)
print images.shape
print X.shape
print labels.shape

# split all pixels into training and testing sets
num_pixels = X.shape[0]
num_pixel_train = round(num_pixels * trainRatio)
print('num_pixels: ' + str(num_pixels) + ' num_pixel_train: ' + str(num_pixel_train))

xTrain = X[:num_pixel_train]
xTest = X[num_pixel_train:]

labelsTrain = labels[:num_pixel_train]
labelsTest = labels[num_pixel_train:]
np.save('./out/true_label.npy', labelsTest)

# phi = (theta, tau)
thetaU = np.random.randint(maxOffset, size=(num_features,2)) # each row is (u1,u2)
thetaV = np.random.randint(maxOffset, size=(num_features,2)) # each row is (v1,v2)
tau = np.random.rand(num_features)

# feature mapping
if mapped:
  features = map_features(xTrain, thetaU, thetaV, images)
  np.save(feature_train_path, features)
else:
  features = np.load(feature_train_path)
print(features.shape)

# train a random forest classifier
rf = RandomForestClassifier(n_estimators=3, criterion='entropy', max_depth=20)
rf.fit(features, labelsTrain)

# prediction
if mapped:
  test_features = map_features(xTest, thetaU, thetaV, images)
  np.save(feature_test_path, test_features)
else:
  test_features = np.load(feature_test_path)
print(test_features.shape)

pred_label = rf.predict(test_features)
pred_prob = rf.predict_proba(test_features)
np.save('./out/pred_label_1000.npy', pred_label)
np.save('./out/pred_prob_1000.npy', pred_prob)

# accuracy
accuracy = 0.0
accClass = np.zeros((numJoints,2))
for i in range(0,pred_label.shape[0]):
    accClass[labelsTest[i]][1] += 1
    print(str(i) + 'th test data label & prob: ' + str(pred_label[i]) + ' ' + str(pred_prob[i]))
    if pred_label[i] == labelsTest[i]:
        accClass[pred_label[i]][0] += 1
        accuracy += 1
accuracy /= float(pred_label.shape[0])

for i in range(0,numJoints):
    print('Class ' + str(i) + ': ' + str(accClass[i][0]) + '/' + str(accClass[i][1]))
print('Total accuracy: ' + str(accuracy))  

score = rf.score(test_features, labelsTest)
print('Score: ' + str(score))

# body parts to joints
joints = part_to_joint(xTest, labelsTest, pred_prob, numData, numJoints)
print('Joints: ')
print(joints.shape)

