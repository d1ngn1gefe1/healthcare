#import cv2
from PIL import Image, ImageFont, ImageDraw
import numpy as np

imageType = 'd'
depth = True

# Read in labels
txt = open('filtered_label.txt', 'r')
labels = np.array([])
for line in txt:
  line = line.split('\n')[0]
  labels = np.append(labels, float(line))

# Add text to each image
font1 = ImageFont.truetype('/Library/Fonts/Arial.ttf', 10)
font2 = ImageFont.truetype('/Library/Fonts/Arial.ttf', 15)
text1 = 'Threshold'
text2 = 'Hand Hygiene!'

num = labels.shape[0]
num1 = 2109
num2 = 4377
offset1 = 14769
offset2 = 30645

count = 0
for i in range(0,num1+num2):
  if i < num1:
    index = 3*i + offset1
  else:
    index = 3*(i-num1) + offset2
  
  if depth:
    img = Image.open(imageType + '_19_21/' + imageType + '-' + str(index) + '.jpg').convert('RGB')
  else:
    img = Image.open(imageType + '_19_21/' + imageType + '-' + str(index) + '.jpg')

  draw = ImageDraw.Draw(img, 'RGBA')
  w1, h1 = draw.textsize(text1, font1)
  w2, h2 = draw.textsize(text2, font2)

  draw.rectangle([(10, 5), (170, 30+h1+h2)], fill=(50, 50, 50, 200))
  draw.text((15, 10), text1, font=font1, fill=(200, 200, 200))
    
  if labels[i] == 1:
    print(count)
    draw.rectangle([(15, 15+h1), (165, 30+h1)], fill=(255, 0, 0))
    draw.text((15+75-w2/2, 30+h1), text2, font=font2, fill=(255, 0, 0))
  else:
    draw.rectangle([(15, 15+h1), (15, 30+h1)], fill=(200, 200, 200))

  draw.rectangle([(15, 15+h1), (165, 30+h1)], outline=(100, 100, 100))    
  img.save(imageType + '_hand_text_19_21/' + str(count) + '.png', 'PNG')
  count += 1
