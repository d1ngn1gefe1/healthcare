#import cv2
from PIL import Image, ImageFont, ImageDraw
import numpy as np

# Read in labels
txt = open('filtered_label.txt', 'r')
labels = np.array([])
for line in txt:
  line = line.split('\n')[0]
  labels = np.append(labels, float(line))

# Add text to each image
font1 = ImageFont.truetype('/Library/Fonts/Arial.ttf', 10)
font2 = ImageFont.truetype('/Library/Fonts/Arial.ttf', 20)
text1 = 'Threshold'
text2 = 'Hand Hygiene!'

num = labels.shape[0]
num1 = 2109
num2 = 4377
offset1 = 14769
offset2 = 30645

count = 0
for i in range(0,num1):
  index = 3*i + offset1
  img = Image.open('d_19_21/d-' + str(index) + '.jpg')

  draw = ImageDraw.Draw(img)
  #draw.rectangle([(10, 10), (170, 70)], fill=(200, 200, 200, 100))
  draw.line([(90, 15), (90, 30)], fill=(200, 0, 0))
  draw.rectangle([(15, 15), (165, 30)], outline=(100, 100, 100))
  w1, h1 = draw.textsize(text1, font1)
  draw.text((15+75-w1/2, 30), text1, font=font1, fill=(100, 0, 0))

  if labels[count] == 1:
    print(count)
    draw.rectangle([(15, 15), (165, 30)], fill=(100, 0, 0))
    w2, h2 = draw.textsize(text2, font2)
    draw.text((15+75-w2/2, 35+h1), text2, font=font2, fill=(100, 0, 0))
    
  img.save('d_hand_text_19_21/' + str(count) + '.jpg', 'JPEG', quality=90)
  count += 1

for i in range(0,num2):
  index = 3*i + offset2
  img = Image.open('d_19_21/d-' + str(index) + '.jpg')

  draw = ImageDraw.Draw(img)
  #draw.rectangle([(10, 10), (170, 70)], fill=(200, 200, 200, 100))
  draw.line([(90, 15), (90, 30)], fill=(200, 0, 0))
  draw.rectangle([(15, 15), (165, 30)], outline=(100, 100, 100))
  w1, h1 = draw.textsize(text1, font1)
  draw.text((15+75-w1/2, 30), text1, font=font1, fill=(100, 0, 0))

  if labels[count] == 1:
    print(count)
    draw.rectangle([(15, 15), (165, 30)], fill=(100, 0, 0))
    w2, h2 = draw.textsize(text2, font2)
    draw.text((15+75-w2/2, 35+h1), text2, font=font2, fill=(100, 0, 0))    

  img.save('d_hand_text_19_21/' + str(count) + '.jpg', 'JPEG', quality=90)
  count += 1


