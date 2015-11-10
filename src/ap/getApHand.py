import numpy as np
from sklearn.metrics import average_precision_score

labelDir = '/scail/scratch/group/vision/bypeng/healthcare/data/'
predLabelFile = 'predLabels_rgb_crop.txt'
trueLabelFile = 'hh_rgb_label_crop.txt'

predLabel = open(labelDir + predLabelFile, 'r')
trueLabel = open(labelDir + trueLabelFile, 'r')

yPred = np.array([])
yTrue = np.array([])

for line in predLabel:
  line = line.split('\n')[0]
  yPred = np.append(yPred, float(line))
for line in trueLabel:
  line = line.split('\n')[0]
  yTrue = np.append(yTrue, float(line))

print average_precision_score(yTrue, yPred)
