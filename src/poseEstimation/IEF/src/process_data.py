import cv2
import numpy as np
from os import listdir, path
from scipy.stats import multivariate_normal
import h5py

data_dir = '../data/'
lsp_ext_dir = '../data/lsp_ext/'
image_path = lsp_ext_dir + 'lsp_images/' 
img_width = 224
img_height = 224
num_channel = 3
num_image = 10000
num_joints = 14
out_image_path = lsp_ext_dir + 'images.npy'
out_joint_path = lsp_ext_dir + 'joints.npy'
cov = [[1e-2, 0], [0, 1e-2]]

def process_image():
  images = []

  for image in listdir(image_path):
    if (image.find('.jpg') != -1):
      img = cv2.imread(image_path + image)
      img = cv2.resize(img, (img_width, img_height))
      images.append(img.reshape(num_channel, img_width, img_height))

  images = np.array(images)
  return images

def process_joint():
  joints = np.loadtxt(joint_path, delimiter=',')[:,:2]
  joints = joints.reshape(num_image, num_joints, 2)
  return joints

def npy_to_h5(images, labels, train_flag):
  labels = labels.reshape(labels.shape[0], num_joints*2)

  if train_flag:
    f = h5py.File(data_dir + 'train.h5', 'w')
  else:
    f = h5py.File(data_dir + 'test.h5', 'w')
  f.create_dataset('data_ief', data=images) 
  f.create_dataset('label_ief', data=labels)  
  f.close()

# joints is (28,) or (14,2) np array
# output: heat_maps (num_joints x 224 x 224)
def joint_to_hm(joints, num_joints):
  joints = joints.reshape(num_joints, 2)
  hm_shape = np.ones((img_width, img_height))
  pair = np.nonzero(hm_shape)
  hm_index = np.array(zip(pair[0],pair[1])).reshape(img_width, img_height, 2)
  heat_maps = []
  for i in range(num_joints):
    mean = joints[i]
    hm = multivariate_normal.pdf(hm_index, mean, cov)
    scale = np.amax(hm) - np.amin(hm)
    hm /= scale
    heat_maps.append(hm)
  heat_maps = np.array(heat_maps)
  return heat_maps

def get_bounded_correction(y, yt, L=20):
  u = y - yt # 1600 x 14 x 2
  u_norm = np.sqrt(np.sum(u**2, axis=2)).reshape(u.shape[0], u.shape[1], 1) 
  unit = u / u_norm
  correction = np.zeros(unit.shape)
  c_norm = np.minimum(L * np.ones(u_norm.shape), u_norm)
  correction[:,:,0] = unit[:,:,0] * c_norm.reshape(unit.shape[:2])
  correction[:,:,1] = unit[:,:,1] * c_norm.reshape(unit.shape[:2])
  return correction

# images: N x 3 x 224 x 224
# yt: N x 14 x 2
def add_hms(images, yt, num_joints):
  out = []
  N = images.shape[0]
  for n in range(N):
    image = images[n]
    hms = joint_to_hm(yt[n], num_joints)
    out_n = np.vstack((image, hms))
    out.append(out_n)
  out = np.array(out)
  print 'add_hms done!'
  return out

# input: y_true (400 x 14 x 2), y_pred(400 x 14 x 2)
# output: correct_count (14 x 1) -- for each joint number of correct predictions
def eval_accuracy(y_true, y_pred, threshold=10):
  '''
  right_leg = np.sqrt(np.sum((y_true[:,0,:] - y_true[:,1,:])**2, axis=1))
  right_thigh = np.sqrt(np.sum((y_true[:,1,:] - y_true[:,2,:])**2, axis=1))
  left_thigh = np.sqrt(np.sum((y_true[:,4,:] - y_true[:,3,:])**2, axis=1))
  left_leg = np.sqrt(np.sum((y_true[:,5,:] - y_true[:,4,:])**2, axis=1))
  right_arm1 = np.sqrt(np.sum((y_true[:,7,:] - y_true[:,6,:])**2, axis=1))
  right_arm2 = np.sqrt(np.sum((y_true[:,8,:] - y_true[:,7,:])**2, axis=1))
  left_arm2 = np.sqrt(np.sum((y_true[:,10,:] - y_true[:,9,:])**2, axis=1))
  left_arm1 = np.sqrt(np.sum((y_true[:,11,:] - y_true[:,10,:])**2, axis=1))
  neck = np.sqrt(np.sum((y_true[:,13,:] - y_true[:,12,:])**2, axis=1))
  '''

  num_images = y_true.shape[0]
  num_joints = y_true.shape[1]
  diff = np.sqrt(np.sum((y_true - y_pred)**2,axis=2))
  th_mask = np.ones(diff.shape) * threshold
  correct_count = np.sum(np.array(diff < th_mask, dtype=int), axis=0) 
  accuracy = correct_count / float(num_images)
  avg_acc = np.mean(accuracy)
  return (correct_count, accuracy, avg_acc)

def main():
  image_saved = path.isfile(out_image_path)
  joint_saved = path.isfile(out_joint_path)

  if image_saved and joint_saved:
    images = np.load(out_image_path)
    joints = np.load(out_joint_path)
  else:
    images = process_image()
    joints = process_joint()
    np.save(out_image_path, images)
    np.save(out_joint_path, joints)

  # npy_to_h5(images, joints)

if __name__ == "__main__":
  main()
