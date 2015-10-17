require 'cutorch'
require 'nn'
require 'image'

-- READ TRAIN IMAGES

imageType = 'rgb'

trainFramesDir = '/scail/data/group/vision/u/syyeung/hospital/data/train/frames/' .. imageType .. '/'
trainLabelsFile = '/scail/data/group/vision/u/syyeung/hospital/data/train/labels.txt'
files = {}
for file in paths.files(trainFramesDir) do
   if file:find('jpg' .. '$') then
      table.insert(files, paths.concat(trainFramesDir,file))
   end
end

trainFiles = {}
numFiles = #files
for i = 0,numFiles-1 do
    local fileName = trainFramesDir .. imageType .. '-' .. i .. '.jpg'
    table.insert(trainFiles, fileName)
end

trainLabels = {}
file = io.open(trainLabelsFile)
if file then
    for line in file:lines() do
        local label = tonumber(line)
        table.insert(trainLabels, label)
    end
else
end
-- HACK: add 1 extra because we had a mismatch during annotation
table.insert(trainLabels, 1)

-- read images
trainImages = {}
for i,file in ipairs(trainFiles) do
   table.insert(trainImages, image.load(file))
end
-- Display a of few them
-- for i = 1,math.min(#trainFiles,10) do
--    image.display{image=trainImages[i], legend=trainFiles[i]}
-- end

image_size = trainImages[1]:size()

-- READ TEST IMAGES

testFramesDir = '/scail/data/group/vision/u/syyeung/hospital/data/test/frames/' .. imageType ..'/'
testLabelsFile = '/scail/data/group/vision/u/syyeung/hospital/data/test/labels.txt'
files = {}
for file in paths.files(testFramesDir) do
   if file:find('jpg' .. '$') then
      table.insert(files, paths.concat(testFramesDir,file))
   end
end

testFiles = {}
numFiles = #files
for i = 0,numFiles-1 do
    local fileName = testFramesDir .. imageType .. '-' .. i .. '.jpg'
    table.insert(testFiles, fileName)
end

testLabels = {}
file = io.open(testLabelsFile)
if file then
    for line in file:lines() do
        local label = tonumber(line)
        table.insert(testLabels, label)
    end
else
end
-- HACK: add 2 extra because we had a mismatch during annotation
table.insert(testLabels, 1)
table.insert(testLabels, 1)

-- read images
testImages = {}
for i,file in ipairs(testFiles) do
   table.insert(testImages, image.load(file))
end

-- create and save data object

hh = {}
hh.train = {}
hh.train.data = torch.Tensor(#trainImages, image_size[1], image_size[2], image_size[3])
for i = 1,#trainImages do
    if i % 100 == 0 then
        print('processing train image ' .. i)
    end
    hh.train.data[i] = trainImages[i]
end
hh.train.labels = torch.Tensor(#trainImages)
for i = 1,#trainImages do
    hh.train.labels[i] = trainLabels[i]
end

hh.test = {}
hh.test.data = torch.Tensor(#testImages, image_size[1], image_size[2], image_size[3])
for i = 1,#testImages do
    if i % 100 == 0 then
        print('processing test image ' .. i)
    end
    hh.test.data[i] = testImages[i]
end
hh.test.labels = torch.Tensor(#testImages)
for i = 1,#testImages do
    hh.test.labels[i] = testLabels[i]
end

dataFile = '../data/hh_' .. imageType .. '.t7'
torch.save(dataFile, hh)
