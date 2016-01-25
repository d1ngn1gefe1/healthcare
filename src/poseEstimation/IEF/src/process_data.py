import cv2
import numpy as np
from os import listdir
# import h5py

data_dir = '../data/'
image_path = data_dir + 'lsp_images/' 
img_width = 224
img_height = 224
num_channel = 3
num_image = 2000
num_joints = 14
joint_path = data_dir + 'joints.txt'
out_image_path = data_dir + 'images.npy'
out_joint_path = data_dir + 'joints.npy'

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
  

images = process_image()
joints = process_joint()
np.save(out_image_path, images)
np.save(out_joint_path, joints)
