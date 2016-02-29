import numpy as np
from util import *

# true_joints: 500 x 14 x 3
# output: 14 x 1 (accuracy for each joint)
def get_joint_acc(true_joints, pred_joints, true_joint_count, pixel_thred=100, offset=10):
  # if a joint has less than 100 pixels, count as occluded 
  mask = np.array(true_joint_count > pixel_thred, dtype=int) # (300, 8)
  true_num_image = np.sum(mask, axis=0) # (8,) 
  diff = abs(true_joints - pred_joints) / 10 # convert mm to cm
  dist = np.sum(np.sqrt(diff), axis=2)
  result = np.array(dist < offset, dtype=int) * mask
  result = np.sum(result, axis=0)
  accuracy = result.astype(float) / true_num_image 
  print true_num_image
  return accuracy

def pixel2world2(coord, W=320, H=240, C=3.51e-3):
  coordW = np.zeros(coord.shape)
  coordW[:,0] = (coord[:,0] - 0.5 * W) * coord[:,2] * C
  coordW[:,1] = -(coord[:,1] - 0.5 * H) * coord[:,2] * C
  coordW[:,2] = coord[:,2]
  return coordW

def get_joint_count(X, labels, num_data, offset, num_joints):
  joint_count = np.zeros((num_data, num_joints))
  for i in range(num_data):
    start = np.where(X[:,3] == i + offset)[0][0]
    end = np.where(X[:,3] == i + offset)[0][-1]
    for j in range(num_joints):
      joint_count[i, j] = len(np.where(labels[start:end] == j)[0])
  return joint_count

def gaussian_density(X, pred_prob, num_joints, b=0.065, push_back=0.039):
  new_prob = np.zeros(pred_prob.shape)
  depth = X[:,2].reshape(len(X),1) / 1000 + push_back# mm -> m
  new_prob = pred_prob * depth**2 
  coord_world = pixel2world2(X[:,:3])
  density = np.zeros(pred_prob.shape)
  for i in range(len(coord_world)):
    density[i] = np.sum(new_prob * np.exp(-np.sum(((coord_world - coord_world[i])/1000/b)**2, axis=1)).reshape(len(new_prob),1), axis=0)
    density[i] /= np.sum(density[i])
  return density

def main():
  root_dir = '/mnt0/emma/shotton/data_ext/'
  test_data = root_dir + 'data1/03/'
  out_prob = root_dir + 'out/ensemble1_prob_3_1500.npy'
  out_label = root_dir + 'out/ensemble1_label_3_1500.npy'
  density_path = test_data + 'density_1500.npy'
  num_images = 300
  offset = 1500
  num_joints = 8

  X = np.load(test_data + 'X.npy')
  true_label = np.load(test_data + 'labels.npy')
  # true_joints: x_world, y_world, z, x_pixel, y_pixel
  true_joints = np.load(test_data + 'joints.npy')[:,:num_joints,:3]
  pred_prob, pred_label = np.load(out_prob), np.load(out_label)
  pred_joints = part_to_joint(X, pred_label, pred_prob, num_images, offset, num_joints, density_path)
  np.save(test_data + 'pred_joints.npy', pred_joints)

  true_joint_count = get_joint_count(X, true_label, num_images, offset, num_joints)
  total_acc = get_joint_acc(true_joints, pred_joints, true_joint_count)
  avg_acc = np.mean(total_acc)
  print 'Estimate from', density_path
  print 'Precision per joint:', total_acc
  print 'Mean average accuracy:', avg_acc
  print '###########################################'

if __name__ == "__main__":
  main()
