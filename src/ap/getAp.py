import numpy as np
from sklearn.metrics import average_precision_score

imageTypes = ['rgb']
directory = '/scail/scratch/group/vision/hospital/'
crop = True
random = False
baseline = False

labelsFile = directory + 'src/ap/'
for i in range(0, len(imageTypes)):
    labelsFile = labelsFile + imageTypes[i] + '_'
if crop:
    labelsFile = labelsFile + 'crop_'
if random:
    labelsFile = labelsFile + 'random_'
if baseline:
    labelsFile = labelsFile + 'baseline_'
pfix = ['test_scores.txt', 'test_true.txt', 'train_scores.txt', 'train_true.txt']
#pfix = ['train_scores.txt', 'train_true.txt']
for i in range(0, 1):
    file1 = labelsFile + pfix[i*2]
    file2 = labelsFile + pfix[i*2+1]
    print(file1)
    print(file2)
    scores = open(file1, 'r')
    trueLabels = open(file2, 'r')
    yTrue = np.array([])
    yScores = np.array([])
    for line in scores:
        line = line.split('\n')[0]
        yScores = np.append(yScores, float(line))
    for line in trueLabels:
        line = line.split('\n')[0]
        yTrue = np.append(yTrue, float(line))
    print average_precision_score(yTrue, yScores)
