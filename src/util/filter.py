import numpy as np

txt = open('predLabels_rgb_crop_19_21.txt', 'r')
filtered = open('filtered_label.txt', 'w')

labels = np.array([])
for line in txt:
  line = line.split('\n')[0]
  labels = np.append(labels, float(line))

num = labels.shape[0]
for i in range(0,num):
  if i > 0 and i < num-1:
    if labels[i-1] == 0 and labels[i+1] == 0:
      labels[i] = 0
  filtered.write(str(labels[i])+'\n')
