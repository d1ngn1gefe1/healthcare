import numpy as np
from sklearn.metrics import average_precision_score

# parameters
fileName = 'rgb_crop_19_21'
directory = '/scail/scratch/group/vision/hospital/src/ap/'
# end parameters

labelsFile = directory + fileName + '_'

pfix = ['test_smooth_scores.txt', 'test_true.txt', 'train_scores.txt', 'train_true.txt']
deleteList = [(200, 630), (660, 690), (720, 1099), (1170, 2807), (3110, 3565), (3592, 6485)] #inclusive

timeLimit = 50
fps = 5

for i in range(0, 2):
    file1 = labelsFile + pfix[i*2]
    file2 = labelsFile + pfix[i*2+1]
    print(file1)
    print(file2)
    scores = open(file1, 'r')
    trueLabels = open(file2, 'r')
    yTrue = np.array([])
    yScores = np.array([])
    j = 0
    for line in scores:
        line = line.split('\n')[0]
        drop = False
        for interval in deleteList:
            if j >= interval[0] and j <= interval[1]:
                drop = True
                break
        if not drop:
            yScores = np.append(yScores, float(line))
        j = j + 1
        if j > timeLimit*fps:
            break

    j = 0
    for line in trueLabels:
        line = line.split('\n')[0]
        drop = False
        for interval in deleteList:
            if j >= interval[0] and j <= interval[1]:
                drop = True
                break
        if not drop:
            yTrue = np.append(yTrue, float(line))
        j = j + 1
        if j > timeLimit*fps:
            break
    print average_precision_score(yTrue, yScores)
