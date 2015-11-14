import os

dirPath = 'd_hand_text_19_21'

my_files = [f for f in os.listdir(dirPath) if f[0].isdigit() and f.endswith(".jpg")]
sorted_files = sorted(my_files,key=lambda x:int(x.split(".")[0])) # sort the file names by starting number

for i,fn in enumerate(sorted_files,1):
  os.rename(dirPath + '/' + sorted_files[i-1], dirPath + '/' + "{0:d}.jpg".format(i-1))
