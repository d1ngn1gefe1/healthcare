import numpy as np
from sklearn.metrics import confusion_matrix
import util

data_saved = True
out_path = './out/'

def get_accuracy():
  # Get average classification accuracy (average of diagoanl of confusion matrix)
  pred_label = np.load(out_path + 'pred_label_1000.npy')
  true_label = np.load(out_path + 'true_label.npy')

  avg_accuracy = util.get_cm_acc(true_label, pred_label)
  return avg_accuracy

# Get inferred joint locations
def get_joints():
  if data_saved:
    data = np.load(out_path + 'matrix.npy')
  else:
    data = util.load_data()
  pred_prob = np.load(out_path + 'pred_prob_1000.npy')
  joints = util.get_joints(data, pred_prob, num_joints, pixel_per_joint, train_ratio)
  np.save('./out/joints_1000.npy', joints)
  return joints
