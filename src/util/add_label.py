from PIL import Image, ImageFont, ImageDraw
import numpy as np
import sys
import os

assert(len(sys.argv) == 2), 'arg: depth (t, f)'
depth = True if sys.argv[1] == 't' else 'f'

dir = '/Users/alan/Documents/research/dataset/new/'
origDir = '/scail/data/group/vision/u/syyeung/hospital/data/'
fileName = 'rgb_crop_19_21'

labelsPath = dir + 'ap/' + fileName + '_test_smooth_predict.txt'
probPath = dir + 'ap/' + fileName + '_test_smooth_scores.txt'
testPath = dir + 'ap/' + fileName + '_test_path.txt'
outDir = dir + ('d/' if depth else 'rgb/')
if not os.path.exists(outDir):
    os.makedirs(outDir)

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

font1 = ImageFont.truetype('/Library/Fonts/Arial.ttf', 11)
font2 = ImageFont.truetype('/Library/Fonts/Arial.ttf', 15)
text1 = 'Threshold'
text2 = 'Hand Hygiene!'

num = labels.shape[0]
for i in range(0, num):
    index = 3*i
    if (depth):
        test[i] = test[i].replace('rgb', 'd')
        img = Image.open(dir + test[i]).convert('RGB')
    else:
        img = Image.open(dir + test[i]).convert('RGB')
    #print(dir + test[i])
    assert (img != None)

    draw = ImageDraw.Draw(img, 'RGBA')
    w1, h1 = draw.textsize(text1, font1)
    w2, h2 = draw.textsize(text2, font2)

    draw.rectangle([(10, 5), (170, 30+h1+h2)], fill=(50, 50, 50, 200))
    draw.text((15, 10), text1, font=font1, fill=(200, 200, 200))
    
    if labels[i] == 1:
        draw.rectangle([(15, 15+h1), (15+150*prob[i], 30+h1)], fill=(255, 0, 0))
        draw.text((15+75-w2/2, 30+h1), text2, font=font2, fill=(255, 0, 0))
    else:
        draw.rectangle([(15, 15+h1), (15+150*prob[i], 30+h1)], fill=(200, 200, 200))

    draw.rectangle([(15, 15+h1), (165, 30+h1)], outline=(100, 100, 100))
    img.save(outDir + str(i) + '.png', 'PNG')
    #print(outDir + test[i].replace('/', '-'))
