import numpy as np
from sklearn.ensemble import RandomForestClassifier
from util import load_data, processData

# parameters
numJoints = 14
pixelPerJoint = 140
trainRatio = 0.7
out_path = './out/'
data_file = out_path + 'matrix.npy'
feature_path = out_path + 'features_all.npy'

data_saved = True

# load data
if not data_saved:
    data = load_data()
else:
    data = np.load(data_file)
    print('Data loaded!')

# data = data[:100]
numData = data.shape[0]
print('Num data: ' + str(numData))

(images, X, labels) = processData(data, numJoints, pixelPerJoint)
print images.shape
print X.shape
print labels.shape

# load features
features = np.load(feature_path)

# split all features into training and testing sets
num_pixels = features.shape[0]
num_pixel_train = round(num_pixels * trainRatio)
print('num_pixels: ' + str(num_pixels) + ' num_pixel_train: ' + str(num_pixel_train))

feature_train = features[:num_pixel_train]
feature_test = features[num_pixel_train:]

labels_train = labels[:num_pixel_train]
labels_test = labels[num_pixel_train:]
np.save(out_path + 'true_label.npy', labels_test)

# train a random forest classifier
rf = RandomForestClassifier(n_estimators=3, criterion='entropy', max_depth=20)
rf.fit(feature_train, labels_train)

# prediction
pred_label = rf.predict(feature_test)
pred_prob = rf.predict_proba(feature_test)
np.save(out_path + 'pred_label_1000.npy', pred_label)
np.save(out_path + 'pred_prob_1000.npy', pred_prob)

# accuracy
accuracy = 0.0
accClass = np.zeros((numJoints,2))
for i in range(0,pred_label.shape[0]):
    accClass[labels_test[i]][1] += 1
    print(str(i) + 'th test data label & prob: ' + str(pred_label[i]) + ' ' + str(pred_prob[i]))
    if pred_label[i] == labels_test[i]:
        accClass[pred_label[i]][0] += 1
        accuracy += 1
accuracy /= float(pred_label.shape[0])

for i in range(0,numJoints):
    print('Class ' + str(i) + ': ' + str(accClass[i][0]) + '/' + str(accClass[i][1]))
print('Total accuracy: ' + str(accuracy))  

score = rf.score(feature_test, labels_test)
print('Score: ' + str(score))
