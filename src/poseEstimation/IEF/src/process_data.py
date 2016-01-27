import cv2
import numpy as np
from os import listdir, path
from scipy.stats import multivariate_normal
import h5py

data_dir = '../data/'
image_path = data_dir + 'lsp_images/' 
img_width = 224
img_height = 224
num_channel = 3
num_image = 2000
num_joints = 14
train_ratio = 0.8
joint_path = data_dir + 'joints.txt'
out_image_path = data_dir + 'images.npy'
out_joint_path = data_dir + 'joints.npy'
cov = [[1, 0], [0, 1]]

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

def npy_to_h5(images, joints):
  joints = joints.reshape(num_image, num_joints*2)
  num_total = images.shape[0]
  num_train = round(num_total * train_ratio)
  train_images = images[:num_train]
  train_joints = joints[:num_train]
  test_images = images[num_train:]
  test_joints = joints[num_train:]

  f = h5py.File(data_dir + 'train.h5', 'w')
  f.create_dataset('data', data=train_images) 
  f.create_dataset('label', data=train_joints)  
  f.close()

  f = h5py.File(data_dir + 'test.h5', 'w')
  f.create_dataset('data', data=test_images) 
  f.create_dataset('label', data=test_joints)  
  f.close()

# joints is (28,) or (14,2) np array
# output: heat_maps (num_joints x 224 x 224)
# bug: all prob are 0 when cov = [[1,0],[0,1]]
def joint_to_hm(joints, num_joints):
  joints = joints.reshape(num_joints, 2)
  hm_shape = np.ones((img_width, img_height))
  pair = np.nonzero(hm_shape)
  hm_index = np.array(zip(pair[0],pair[1])).reshape(img_width, img_height, 2)
  heat_maps = []
  for i in range(num_joints):
    mean = joints[i]
    hm = multivariate_normal.pdf(hm_index, mean, cov)
    heat_maps.append(hm)
  heat_maps = np.array(heat_maps)
  return heat_maps

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

  npy_to_h5(images, joints)

if __name__ == "__main__":
  main()
