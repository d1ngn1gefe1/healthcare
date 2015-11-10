dispenserLoc = {32, 64}
threshold = 45 
cropSize = 32
stride = 4
imageNum = 3686

hh = torch.load('/scail/scratch/group/vision/hospital/data/hh_rgb_crop.t7')

trueLabels = hh.test.labels
dataFile1 = 'handDetect_1-700_' .. cropSize .. '.t7'
dataFile2 = 'handDetect_701-1400_' .. cropSize .. '.t7'
dataFile3 = 'handDetect_1401-1958_' .. cropSize .. '.t7'

dataFile = 'handDetect_crop_' .. cropSize .. '.t7' 

imageLabels1 = torch.load('/scail/scratch/group/vision/bypeng/healthcare/src/' .. dataFile1)
imageLabels2 = torch.load('/scail/scratch/group/vision/bypeng/healthcare/src/' .. dataFile2)
imageLabels3 = torch.load('/scail/scratch/group/vision/bypeng/healthcare/src/' .. dataFile3)

--imageLabels = {}
--for i = 1,700 do
--  imageLabels[i] = imageLabels1[i]
--end

--for i = 701,1400 do
--  imageLabels[i] = imageLabels2[i]
--end

--for i = 1401,1958 do
--  imageLabels[i] = imageLabels3[i]
--end

imageLabels = torch.load('/scail/scratch/group/vision/bypeng/healthcare/src/' .. dataFile)

imageWidth = 64
imageHeight = 64
numX = (imageWidth-cropSize)/stride
numY = (imageHeight-cropSize)/stride

handLocations = {}
for n = 1, #imageLabels do
  local image = imageLabels[n]
  local indices = {}
  local j = 1
  for i = 1, #image do
    if image[i] == 2 then
      x = i % numX 
      if x == 0 then
        x = numX
      end	
      y = math.ceil(i / numY)
      x = x * stride + cropSize
      y = y * stride + cropSize
      indices[j] = {x, y}
      j = j + 1
    end
  end
  handLocations[n] = indices
end

labels = {}
for n = 1, #handLocations do
  indices = handLocations[n]
  if #indices == 0 then
    labels[n] = 0
  else
    min = 240
    for j =1, #indices do
      dist = math.sqrt((indices[j][1]-dispenserLoc[1])^2 + (indices[j][2]-dispenserLoc[2])^2)
      print('Image: ' .. n .. ' Dist: ' .. dist)
      if dist < min then
        min = dist
      end
    end 
    if min < threshold then
      labels[n] = 1
    else 
      labels[n] = 0
    end
  end
end

predLabels = io.open('predLabels_rgb_crop.txt', 'w')
trueLabelsFile = io.open('hh_rgb_label_crop.txt', 'w')

correct = 0
correct_0 = 0
correct_1 = 0
total_0 = 0
total_1 = 0
sum = 0
for i = 1, #labels do
  predLabels:write(labels[i] .. '\n')
  trueLabelsFile:write(trueLabels[i] .. '\n')
  if trueLabels[i] == 1 then
    total_1 = total_1 + 1
  else
    total_0 = total_0 + 1
  end
  if labels[i] == trueLabels[i] then
    correct = correct + 1
    if labels[i] == 1 then
      correct_1 = correct_1 + 1
    else
      correct_0 = correct_0 + 1
    end
  end
end

accuracy = correct / #handLocations

print('Total image:', #handLocations)
print('Correct:' , correct)
print('Correct 0:' ..  correct_0 .. '/' .. total_0)
print('Correct 1:', correct_1 .. '/' .. total_1)
print('Accracy:', accuracy)
