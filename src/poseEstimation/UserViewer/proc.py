import numpy as np
import argparse
import glob
import cv2
import sys
import os
from sklearn.neighbors import KNeighborsClassifier
np.set_printoptions(threshold=np.nan)

''' order:
  JOINT_HEAD
  JOINT_NECK
  JOINT_LEFT_SHOULDER
  JOINT_RIGHT_SHOULDER
  JOINT_LEFT_ELBOW
  JOINT_RIGHT_ELBOW
  JOINT_LEFT_HAND
  JOINT_RIGHT_HAND
  JOINT_TORSO
  JOINT_LEFT_HIP
  JOINT_RIGHT_HIP
  JOINT_LEFT_KNEE
  JOINT_RIGHT_KNEE
  JOINT_LEFT_FOOT
  JOINT_RIGHT_FOOT
'''

H = 240
W = 320
nJoints = 15
nPeople = 20

palette = [(34, 69, 101), (0, 195, 243), (146, 86, 135), (130, 132, 132),\
           (0, 132, 243), (241, 202, 161), (50, 0, 190), (128, 178, 194), \
           (23, 45, 136), (86, 136, 0), (172, 143, 230), (165, 103, 0), \
           (121, 147, 249), (151, 78, 96), (0, 166, 246), (108, 68, 179), \
           (34, 88, 226), (0, 211, 220), (0, 182, 141), (38, 61, 43)] # BGR

jointName = ['HEAD', 'NECK', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', \
             'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_HAND', 'RIGHT_HAND', \
             'TORSO', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', \
             'RIGHT_KNEE', 'LEFT_FOOT', 'RIGHT_FOOT']

skeleton = [(0,1), (1,2), (1,3), (2,4), (3,5), (4,6), (5,7), (2,8), (3,8), \
            (8,9), (8,10), (9,10), (9,11), (10,12), (11,13), (12,14)]
# first joint vs second joint
relativeWeights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, \
                   0.3, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5]

'''
  pixelZ = worldZ
  pixelX = worldX/worldZ/C + W/2.0
  pixelY = -worldY/worldZ/C + H/2.0
'''
def getC(joints):
  C = 0
  for joint in joints:
    C += joint[0]/joint[2]/(joint[3]-W/2.0)
    C += -joint[1]/joint[2]/(joint[4]-H/2.0)
  C /= joints.shape[0]*2
  return C

def pixel2world(pixel, C):
  world = np.empty(pixel.shape)
  world[:, 2] = pixel[:, 2]
  world[:, 0] = (pixel[:, 0]-W/2.0)*C*pixel[:, 2]
  world[:, 1] = -(pixel[:, 1]-H/2.0)*C*pixel[:, 2]
  return world

def world2pixel(world, C):
  pixel = np.empty(world.shape)
  pixel[:, 2] = world[:, 2]
  pixel[:, 0] = world[:, 0]/world[:, 2]/C + W/2.0
  pixel[:, 1] = -world[:, 1]/world[:, 2]/C + H/2.0
  return pixel.astype(int)

def visualizePts(pixel, name, label=None):
  if label is None:
    img = np.zeros((H, W), np.uint8)
  else:
    img = np.zeros((H, W, 3), np.uint8)

  for i, pt in enumerate(pixel):
    if label is None:
      img[pt[1], pt[0]] = pt[2]
    else:
      #img[pt[1], pt[0]] = np.asarray(palette[label[i]])
      cv2.circle(img, (pt[0], pt[1]), 1, palette[label[i]], -1)
      #print label[i], palette[label[i]]
    cv2.imshow(name, img)

def drawID(img, id):
  cv2.putText(img, str(id), (10, 10), \
              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
  return img

def drawJoints(img, joints, bad):
  # right bar
  img = np.hstack((img, np.zeros((H, 100, 3)))).astype(np.uint8)
  for i in range(nJoints):
    cv2.rectangle(img, (W, H*i/nJoints), (W+100, H*(i+1)/nJoints-1), \
                  palette[i], -1)
    cv2.putText(img, jointName[i], (W, H*(i+1)/nJoints-5), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))

  if bad:
    cv2.putText(img, 'bad frame', (20, 20), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))
    return img

  for i, pt in enumerate(joints[:, 3:]):
    cv2.circle(img, tuple(pt.astype(np.uint16)), 4, palette[i], -1)

  for jointPair in skeleton:
    pt1 = joints[jointPair[0], 3:]
    pt2 = joints[jointPair[1], 3:]
    mid = (pt1+pt2)/2

    cv2.line(img, tuple(pt1.astype(np.uint16)), tuple(mid.astype(np.uint16)), \
      palette[jointPair[0]], 2)
    cv2.line(img, tuple(mid.astype(np.uint16)), tuple(pt2.astype(np.uint16)), \
      palette[jointPair[1]], 2)

  return img

def x2yz(pt1, pt2, x):
  y = (x-pt1[0])/(pt2[0]-pt1[0])*(pt2[1]-pt1[1])+pt1[1]
  z = (x-pt1[0])/(pt2[0]-pt1[0])*(pt2[2]-pt1[2])+pt1[2]
  return (np.round(y).astype(int), np.round(z).astype(int))

def y2xz(pt1, pt2, y):
  x = (y-pt1[1])/(pt2[1]-pt1[1])*(pt2[0]-pt1[0])+pt1[0]
  z = (y-pt1[1])/(pt2[1]-pt1[1])*(pt2[2]-pt1[2])+pt1[2]
  return (np.round(x).astype(int), np.round(z).astype(int))

def joints2skeleton(joints):
  ptsAll = np.array([]).reshape(0, 3)
  labelsAll = np.array([]).reshape(0)

  for i, jointPair in enumerate(skeleton):
    x, y, z, nPts = None, None, None, 0
    pt1 = joints[jointPair[0]]
    pt2 = joints[jointPair[1]]

    if abs(pt1[0]-pt2[0]) < abs(pt1[1]-pt2[1]):
      start = min(pt1[1], pt2[1])
      end = max(pt1[1], pt2[1])
      y = np.arange(start, end, (end-start)/20.0)
      x, z = y2xz(pt1, pt2, y)
      nPts = y.shape[0]
      if nPts == 0:
        continue
      labels = np.empty(nPts)
      nPart1 = int(relativeWeights[i]*nPts)
      nPart2 = nPts - nPart1
      nParts = [nPart1, nPart2]
      idxSm = np.argmin([pt1[1], pt2[1]])
      idxLg = np.argmax([pt1[1], pt2[1]])
      labels[:nParts[idxSm]] = jointPair[idxSm]
      labels[nParts[idxSm]:] = jointPair[idxLg]
    else:
      start = min(pt1[0], pt2[0])
      end = max(pt1[0], pt2[0])
      x = np.arange(start, end, (end-start)/20.0)
      y, z = x2yz(pt1, pt2, x)
      nPts = x.shape[0]
      if nPts == 0:
        continue
      labels = np.empty(nPts)
      nPart1 = int(relativeWeights[i]*nPts)
      nPart2 = nPts - nPart1
      nParts = [nPart1, nPart2]
      idxSm = np.argmin([pt1[0], pt2[0]])
      idxLg = np.argmax([pt1[0], pt2[0]])
      labels[:nParts[idxSm]] = jointPair[idxSm]
      labels[nParts[idxSm]:] = jointPair[idxLg]

    pts = np.vstack((x, y, z)).T
    ptsAll = np.concatenate((ptsAll, pts))
    #print ptsAll.shape, pts.shape
    labelsAll = np.concatenate((labelsAll, labels))
    #print labelsAll.shape, labels.shape

  return (ptsAll, labelsAll.astype(int))

# zScale: how much we trust the z value. 1 indicates equal trust on x, y, z
def knn(depth, joints, C, name=None, zScale=1.0):
  pts_world, labels = joints2skeleton(joints[:, :3])
  if name is not None:
      visualizePts(world2pixel(pts_world, C), name, labels)
  pts_world[:, 2] *= zScale
  classifier = KNeighborsClassifier(n_neighbors=5)
  classifier.fit(pts_world, labels)

  X = np.vstack((np.nonzero(depth)[1], np.nonzero(depth)[0]))
  X = np.vstack((X, depth[depth != 0]))
  X_world = pixel2world(X.T, C)
  X_world[:, 2] *= zScale
  predicts = classifier.predict(X_world)

  perPixelLabels = -np.ones(depth.shape)
  perPixelLabels[depth != 0] = predicts

  img = np.zeros((H, W, 3), np.uint8)
  for i in range(nJoints):
    img[perPixelLabels == i] = palette[i]

  return img

def main(**kwargs):
  C = 0
  bad = False
  deleleBad = True
  dataDir = kwargs.get('dir')
  id = kwargs.get('id')
  startFrame = kwargs.get('start')
  key = 0

  for i in range(0 if id == -1 else id, nPeople if id == -1 else id+1):
    person = dataDir+str(i).zfill(2)+'_'+'[0-9]'*5+'_'
    depthSideFiles = sorted(glob.glob(person + 'depth_side.txt'))
    depthTopFiles = sorted(glob.glob(person + 'depth_top.txt'))
    jointsSideFiles = sorted(glob.glob(person + 'joints_side.txt'))
    jointsTopFiles = sorted(glob.glob(person + 'joints_top.txt'))
    labelSideFiles = sorted(glob.glob(person + 'label_side.txt'))
    labelTopFiles = sorted(glob.glob(person + 'label_top.txt'))

    assert len(depthSideFiles) == len(depthTopFiles) == len(jointsSideFiles) \
      == len(jointsTopFiles) == len(labelSideFiles) == len(labelTopFiles)
    N = len(depthSideFiles)
    print 'person id: %d; #frames: %d' % (i, N)

    j = 0
    while j < N:
      curFrame = int(depthSideFiles[j].replace(dataDir, '')\
                   .replace('_depth_side.txt', '').replace(str(i)+'_', ''))
      print 'current frame: %d' % curFrame
      if curFrame < startFrame:
        j += 1
        continue
      if not os.path.exists(depthSideFiles[j]):
        print 'file not exist: %d' % curFrame
        if key == 65361:
          j -= 1
          continue
        else:
          j += 1
          continue

      depthSide = np.loadtxt(depthSideFiles[j], dtype=float).reshape((H, W))
      depthTop = np.loadtxt(depthTopFiles[j], dtype=float).reshape((H, W))
      jointsSide = np.loadtxt(jointsSideFiles[j], dtype=float, delimiter=', ')
      jointsTop = np.loadtxt(jointsTopFiles[j], dtype=float, delimiter=', ')
      labelSide = np.loadtxt(labelSideFiles[j], dtype=int).reshape((H, W))
      labelTop = np.loadtxt(labelTopFiles[j], dtype=int).reshape((H, W))

      if np.count_nonzero(jointsSide) == 0 and \
        np.count_nonzero(jointsSide) == 0:
        bad = True
        if deleleBad:
            print 'automatically delete bad frame %d' % curFrame
            os.remove(depthSideFiles[j])
            os.remove(depthTopFiles[j])
            os.remove(jointsSideFiles[j])
            os.remove(jointsTopFiles[j])
            os.remove(labelSideFiles[j])
            os.remove(labelTopFiles[j])
            continue
      else:
        bad = False

      if C == 0 and (not bad):
        C = getC(jointsSide)

      # make and apply masks
      mostFreqLabel = np.argmax(np.bincount(labelSide.ravel())[1:])+1
      labelSide[labelSide != mostFreqLabel] = 0
      labelSide[labelSide == mostFreqLabel] = 1
      labelTop[labelTop != -1] = 1
      labelTop[labelTop == -1] = 0
      depthSide *= labelSide
      depthTop *= labelTop

      dispSide = cv2.equalizeHist(depthSide.astype(np.uint8))
      dispSide = cv2.applyColorMap(dispSide, cv2.COLORMAP_SPRING)
      dispSide = np.multiply(dispSide, labelSide[:, :, np.newaxis])
      dispSide = drawJoints(dispSide, jointsSide, bad)
      dispSide = drawID(dispSide, curFrame)
      cv2.imshow('depthSide', dispSide)

      dispTop = cv2.equalizeHist(depthTop.astype(np.uint8))
      dispTop = cv2.applyColorMap(dispTop, cv2.COLORMAP_SPRING)
      dispTop = np.multiply(dispTop, labelTop[:, :, np.newaxis])
      dispTop = drawJoints(dispTop, jointsTop, bad)
      dispTop = drawID(dispTop, curFrame)
      cv2.imshow('depthTop', dispTop)

      # knn
      perPixelSide = knn(depthSide, jointsSide, C, 'sideKNN')
      perPixelTop = knn(depthTop, jointsTop, C, 'topKNN')
      cv2.imshow('perPixelSide', perPixelSide)
      cv2.imshow('perPixelTop', perPixelTop)


      key = cv2.waitKey(0)
      if key == ord('s'):
        return
      elif key == 65288:
        print 'deleting frame %d' % curFrame
        os.remove(depthSideFiles[j])
        os.remove(depthTopFiles[j])
        os.remove(jointsSideFiles[j])
        os.remove(jointsTopFiles[j])
        os.remove(labelSideFiles[j])
        os.remove(labelTopFiles[j])
      elif key == 65361:
        j -= 2

      j += 1

      #print depthSide.shape, depthTop.shape, jointsSide.shape, \
      #jointsTop.shape, labelSide.shape, labelTop.shape

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir', required=True)
  parser.add_argument('--id', type=int, default=-1)
  parser.add_argument('--start', type=int, default=0)
  args = parser.parse_args()
  main(**vars(args))
