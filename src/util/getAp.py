import numpy as np
from sklearn.metrics import average_precision_score

# parameters
fileName = 'rgb_crop_19_21'
directory = '/scail/scratch/group/vision/hospital/src/ap/'
# end parameters

labelsFile = directory + fileName + '_'

pfix = ['test_scores.txt', 'test_true.txt', 'train_scores.txt', 'train_true.txt']
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
