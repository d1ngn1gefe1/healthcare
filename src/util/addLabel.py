import cv2
import numpy as np

# Read in labels
txt = open('predLabels_rgb_crop.txt', 'r')
labels = np.array([])
for line in txt:
  line = line.split('\n')[0]
  labels = np.append(labels, float(line))

# Add text to each image
font = cv2.FONT_HERSHEY_COMPLEX
x = 200 #position of text
y = 20 #position of text

num = labels.shape[0] #1406
for i in range(1,num+1):
  index = 3*(i-1)
  img = cv2.imread('d/d-' + str(index) + '.jpg')

  if labels[i-1] == 0:
    cv2.putText(img, "Negative", (x,y), font, 0.6, (255,255,255), 2) #Draw the text
  else:
    cv2.putText(img, "Positive", (x,y), font, 0.8, (255,255,255), 2) #Draw the text
  cv2.imwrite('d_text/' + str(i) + '.jpg', img)

