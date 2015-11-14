import cv2
import numpy as np

dir = '/Users/alan/Documents/research/dataset/new/'
origDir = '/scail/data/group/vision/u/syyeung/hospital/data/'
labelsPath = dir + 'ap/rgb_crop_test_predict.txt'
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

assert (labels.shape[0] == prob.shape[0] == test.shape[0])
num = labels.shape[0]
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
thickness = 1
text1 = 'Probability'
text2 = 'Hand Hygiene!'
textWidth1 = cv2.getTextSize(text1, font, fontScale, thickness)[0][0]
textHeight1 = cv2.getTextSize(text1, font, fontScale, thickness)[0][1]
textWidth2 = cv2.getTextSize(text2, font, fontScale, thickness)[0][0]
textHeight2 = cv2.getTextSize(text2, font, fontScale, thickness)[0][1]
top = 10
left = 20
for i in range(0, num):
    index = 3*i
    img = cv2.imread(dir + test[i])
    #print(dir + test[i])
    assert (img != None)

    cv2.rectangle(img, (left - 5, top - 5), (left + textWidth1 + 110, top + textHeight1 + 5), (200, 200, 200), cv2.FILLED)
    cv2.putText(img, text1, (left, top + textHeight1), font, fontScale, (40, 40, 40), thickness)
    cv2.rectangle(img, (left + textWidth1 + 5, top), (left + textWidth1 + 105, top + textHeight1), (40, 40, 40), cv2.FILLED)
    cv2.rectangle(img, (left + textWidth1 + 5, top), (left + textWidth1 + 5 + int(100*prob[i]), top + textHeight1), (0, 0, 180), cv2.FILLED)

    if labels[i] == 1:
        cv2.putText(img, text2, (left, top + textHeight1 + textHeight2 + 10), font, fontScale, (0, 0, 180))

    cv2.imwrite(outDir + str(i) + '.jpg', img)
    #cv2.imwrite(outDir + test[i].replace('/', '-'), img)
    #print(outDir + test[i].replace('/', '-'))
