import os
import sys
import shutil

assert(len(sys.argv) == 2), 'arg: frames_folder'
dirPath = sys.argv[1]
outPath = dirPath + '/' + 'out'
print(dirPath)
if not os.path.exists(outPath):
    os.makedirs(outPath)

deleteList = [(200, 630), (660, 690), (720, 1099), (1170, 2807), (3110, 3565), (3592, 6485)] #inclusive
images = [f for f in os.listdir(dirPath) if f[0].isdigit() and f.endswith(".png")]
temp = []
for image in images:
	i = int(image.split('.')[0])
	drop = False
	for interval in deleteList:
		if i >= interval[0] and i <= interval[1]:
			drop = True
			break
	if not drop:
		temp.append(image)
		shutil.copy2(dirPath + '/' + image, outPath)
images = temp

sortedImages = sorted(images, key=lambda x:int(x.split(".")[0])) # sort the file names by starting number


for i,fn in enumerate(sortedImages, 1):
	os.rename(outPath + '/' + sortedImages[i-1], outPath + '/' + "{0:d}.png".format(i-1))
