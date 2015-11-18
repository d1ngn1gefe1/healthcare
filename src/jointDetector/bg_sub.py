import cv2
import numpy as np
import os
from os import listdir

def bwareaopen(img, area):
    img2 = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]
    _, contours, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(img.shape, np.uint8)
    maxIdx = -1
    maxArea = 0
    for i in np.arange(len(contours)):
        curArea = cv2.contourArea(contours[i])
        if (curArea >= area and curArea >= maxArea):
        	maxIdx = i
        	maxArea = curArea

    if maxIdx != -1:
    	cv2.drawContours(mask, contours, maxIdx, 255, -1)
    	img = cv2.bitwise_and(img, mask)
    	#img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        #cv2.drawContours(img, contours, maxIdx, (0, 0, 255), 2)
        #cv2.drawContours(img, contours, maxIdx, 255, 2)
    	return img, mask
    else:
    	return mask, mask

def removeZeros(img, mean, lowThred):
	rows = img.shape[0]
	cols = img.shape[1]
	for i in xrange(rows):
		for j in xrange(cols):
			k = img.item(i, j)
			if k <= lowThred:
				img.itemset((i, j), mean[i])
	return img

def createFeatures(img, mask):
	n = np.count_nonzero(mask)
	rows, cols = np.nonzero(mask)
	depth = img[rows, cols]

	features = np.dstack((rows, cols, depth))[0]
	#features[:, 0] = rows
	#features[:, 1] = cols
	            

imgDir = '/Users/alan/Documents/research/dataset/new/cvpr10-18-15morning/d/'
ext = '.jpg'
outDir = imgDir + 'out/'
lowThred = 3
bgRatio = 0.5
meanRatio = 0.7
if not os.path.exists(outDir):
	os.makedirs(outDir)

imgFiles = [f for f in listdir(imgDir) if f.endswith(ext)]
imgFiles.sort(key=lambda x: int(x.split('-')[1][:-len(ext)]))
#print(imgFiles)

tmp = cv2.imread(imgDir + imgFiles[0], 0)
sz = tmp.shape

# calculate mean of each row, and sum of each image
sums = np.zeros(len(imgFiles))
mean = np.zeros(sz[0])
for i, imgFile in enumerate(imgFiles):
	img = cv2.imread(imgDir + imgFile, 0)
	sums[i] = np.sum(img)
	img = np.fliplr(np.sort(img))
	img = img[:,0:sz[1]*meanRatio]
	mean = mean + np.mean(img, axis=1)
indices = np.argsort(sums)[::-1][:len(imgFiles)*bgRatio]
mean = mean/len(imgFiles)

# calculate the background
bg = np.zeros(sz)
for idx in indices:
	img = cv2.imread(imgDir + imgFiles[idx], 0)
	img = removeZeros(img, mean, lowThred)
	bg = bg + img.astype(np.float)
bg = bg/len(indices)
bg = bg.astype(np.uint8)
#print(np.amin(bg), np.amax(bg))
cv2.imwrite(outDir + 'bg.jpg', bg, [cv2.IMWRITE_JPEG_QUALITY, 100])

for imgFile in imgFiles:
	img = cv2.imread(imgDir + imgFile, 0)
	img = removeZeros(img, mean, lowThred)
	img = cv2.subtract(bg, img)
	img = cv2.GaussianBlur(img, (3, 3), 0)
	#element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	#img = cv2.erode(img, element)
	#img = cv2.dilate(img, element)
	#img = cv2.fastNlMeansDenoising(img, None, 5)
	img, mask = bwareaopen(img, 500);
	#createFeatures(img, mask)
	img = cv2.equalizeHist(img)

	#cv2.imshow('img', img)
	#cv2.waitKey()
	cv2.imwrite(outDir + imgFile, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
