import cv2
import numpy as np

dir = '/Users/alan/Documents/research/dataset/new/'
origDir = '/scail/data/group/vision/u/syyeung/hospital/data/'
labelsPath = dir + 'ap/rgb_crop_test_true.txt'
probPath = dir + 'ap/rgb_crop_test_scores.txt'
testPath = dir + 'ap/rgb_crop_test_path.txt'
outDir = dir + 'out/' 
useProb = 1

# Read in labels
labelsFile = open(labelsPath, 'r')
labels = np.array([])
for line in labelsFile:
    line = line.split('\n')[0]
    labels = np.append(labels, float(line))

# Read in probability
probFile = open(probPath, 'r')
prob = np.array([])
for line in probFile:
    line = line.split('\n')[0]
    prob = np.append(prob, float(line))

# Read in test files
testFile = open(testPath, 'r')
test = np.array([])
for line in testFile:
    line = line.split('\n')[0]
    test = np.append(test, line[len(origDir):])

# Add text to each image
font = cv2.FONT_HERSHEY_SIMPLEX
x = 20 # bl corner of text
y = 20 # bl corner of text

assert (labels.shape[0] == prob.shape[0] == test.shape[0])
num = labels.shape[0]
for i in range(0, num):
    index = 3*i
    img = cv2.imread(dir + test[i])
    #print(dir + test[i])
    assert (img != None)

    if labels[i] == 0:
        cv2.putText(img, 'Negative (prob: ' + ('%.3f' % prob[i]) + ')', (x,y), font, 0.5, (255,255,255)) #Draw the text
    else:
        cv2.putText(img, 'Positive (prob: ' + ('%.3f' % prob[i]) + ')', (x,y), font, 0.5, (255,255,255)) #Draw the text

    cv2.imwrite(outDir + str(i) + '.jpg', img)
    #cv2.imwrite(outDir + test[i].replace('/', '-'), img)
    #print(outDir + test[i].replace('/', '-'))
