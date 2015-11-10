require 'cutorch'
require 'nn'
require 'cunn'
require 'image'

-- load images
imageType = 'rgb'
hh = torch.load('/scail/scratch/group/vision/hospital/data/hh_' .. imageType .. '_crop.t7')
images = hh.test.data

height = images:size(3) -- 64
width = images:size(4) -- 64

cropSize = 32
inputSize = 32
stride = 4

saved = false

-- crop and warp each image
if saved == false then
totalCroppedImage = {} -- 1958 * (144 * (3*32*32))
for n = 1, images:size(1) do
  local croppedImage = {} 
  local y = 0
  local k = 1
  for i = 1, (height - cropSize)/stride do
    local x = 0
    for j = 1, (width - cropSize)/stride  do
      croppedImage[k] = image.scale(image.crop(images[n], x, y, x+cropSize, y+cropSize), inputSize, inputSize)
      x = x + stride
      k = k + 1
    end
    y = y + stride
  end
  totalCroppedImage[n] = croppedImage 
end
torch.save('/scail/scratch/group/vision/bypeng/healthcare/data/totalCroppedImage.t7', totalCroppedImage)
else 
totalCroppedImage = torch.load('/scail/scratch/group/vision/bypeng/healthcare/data/totalCroppedImage.t7')
end

print('Each image has:', #(totalCroppedImage[1]))

-- load cnn model
model = torch.load('/scail/scratch/group/vision/bypeng/healthcare/src/handClassifier.bin')

totalImageLabels = {}
for n = 1, images:size(1) do
  print('Image:', n)
  local croppedImageLabel = {}
  local croppedImage = totalCroppedImage[n]
  for i = 1, #croppedImage do
    if (i % 10 == 0) then
      print('Process:', i)
    end
    croppedImage[i] = croppedImage[i]:cuda()
    local prediction = model:forward(croppedImage[i])
    local confidences, indices = torch.sort(prediction, true)
    -- print(indices) -- 1 2
    local label = indices[1]
    croppedImageLabel[i] = label
  end
  totalImageLabels[n] = croppedImageLabel
end

torch.save('/scail/scratch/group/vision/bypeng/healthcare/data/handDetect_crop_' .. cropSize .. '_'  .. imageType .. '.t7', totalImageLabels)
