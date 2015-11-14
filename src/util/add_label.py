from PIL import Image, ImageFont, ImageDraw
import numpy as np

dir = '/Users/alan/Documents/research/dataset/new/'
origDir = '/scail/data/group/vision/u/syyeung/hospital/data/'
fileName = 'rgb_crop_19_21'
depth = true

labelsPath = dir + 'ap/' + fileName + '_test_smooth_predict.txt'
probPath = dir + 'ap/' + fileName + '_test_smooth_scores.txt'
testPath = dir + 'ap/' + fileName + '_test_path.txt'
outDir = dir + 'out/' 

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

font1 = ImageFont.truetype('/Library/Fonts/Arial.ttf', 10)
font2 = ImageFont.truetype('/Library/Fonts/Arial.ttf', 20)
text1 = 'Threshold'
text2 = 'Hand Hygiene!'

num = labels.shape[0]
for i in range(0, num):
    index = 3*i
    img = Image.open(dir + test[i])
    #print(dir + test[i])
    assert (img != None)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(15, 15), (165, 30)], fill=(200, 200, 200))
    draw.rectangle([(15, 15), (15+150*prob[i], 30)], fill=(100, 0, 0))
    draw.line([(90, 15), (90, 30)], fill=(200, 0, 0))
    draw.rectangle([(15, 15), (165, 30)], outline=(100, 100, 100))
    w1, h1 = draw.textsize(text1, font1)
    draw.text((15+75-w1/2, 30), text1, font=font1, fill=(100, 0, 0))
    if labels[i] == 1:
        w2, h2 = draw.textsize(text2, font2)
        draw.rectangle([(15+75-w2/2, 35+h1), (15++75+w2/2, 35+h1+h2)], fill=(200, 200, 200))
        draw.text((15+75-w2/2, 35+h1), text2, font=font2, fill=(100, 0, 0))
    img.save(outDir + str(i) + '.jpg', 'JPEG', quality=90)
    #print(outDir + test[i].replace('/', '-'))
