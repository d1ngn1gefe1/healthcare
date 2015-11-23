import numpy as np
from numpy import genfromtxt, savetxt
from os import listdir

csvDir = '/Users/Emma/Desktop/cvpr10-19-15morning/joint_data/'
imgDir = '/Users/Emma/Desktop/cvpr10-19-15morning/d_19_21/'
ext = 'csv'
pixelPerJoint = 200

def processData():
    files = [f for f in listdir(csvDir) if f.endswith(ext)] 
    X = []
    labels = []
    images = []
    for fileName in files: # e.g. 'd-14808_0.csv'
      label = int(fileName.split('_')[1].split('.csv')[0])
      image = imgDir + fileName.split('_')[0] + '.jpg'
      dataset = genfromtxt(open(csvDir+fileName,'r'), delimiter=',', dtype='f8')[1:]
      numPixel = dataset.shape[0] if dataset.shape[0] < pixelPerJoint else pixelPerJoint
      X = dataset[0:numPixel, 1:4] if X == [] else np.concatenate((X, dataset[0:numPixel, 1:4]))
      labels = np.append(labels, label*np.ones(numPixel))
      images = np.append(images, [image for i in range(0,numPixel)])
    return (X,labels,images)
      
