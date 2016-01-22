import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import util

num_joints = 14
pixel_per_joint = 140
train_ratio = 0.7
label_saved = True
out_path = './out/'

data = np.load(out_path + 'matrix.npy')

# Get average classification accuracy (average of diagoanl of confusion matrix)
if label_saved:
  pred_label = np.load(out_path + 'pred_label_1000.npy')
  true_label = np.load(out_path + 'true_label.npy')
else:
  (X_train, X_test, labels_train, labels_test) = util.partition_data(data, pixel_per_joint, num_joint, train_ratio)

  feature_train = np.load(out_path + 'feature_train_1000.npy')
  feature_test = np.load(out_path + 'feature_test_1000.npy')

  rf = RandomForestClassifier(n_estimators=3, criterion='entropy', max_depth=20)
  rf.fit(feature_train, labels_train)

  true_label = labels_test
  pred_label = rf.predict(feature_test)

avg_accuracy = util.get_cm_acc(true_label, pred_label)
print('Average accuracy: ' + str(avg_accuracy))

# Get inferred joint locations
pred_prob = np.load(out_path + 'pred_prob_1000.npy')
joints = util.get_joints(data, pred_prob, num_joints, pixel_per_joint, train_ratio)
np.save('./out/joints_1000.npy', joints)
print(joints.shape)
