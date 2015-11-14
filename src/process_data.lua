require 'cutorch'
require 'nn'
require 'image'

-- parameters
imageTypes = {'rgb'}
datasets = {'cvpr10-19-15morning', 'cvpr10-21-15'}

coor = {114, 240}
boxSz = 64
crop = true
seed = 1
skip = 3

ratioTest = 0.3 -- the ratio of examples used as the test set (the ratio will be different after dropping negative examples)
setRatioPosTrain = true -- set the ratio of positive examples in training set, by dropping negative examples 
setRatioPosTest = false -- set the ratio of positive examples in test set, by dropping negative examples 
ratioPosTrain = 0.2
ratioPosTest = 0.1

dir = '/scail/scratch/group/vision/hospital/'
dataDir = '/scail/data/group/vision/u/syyeung/hospital/data/'
fileName = 'rgb_crop_19_21'
-- end parameters

dataFile = '../data/' .. fileName .. '.t7'
testPathFile = dir .. 'src/ap/' .. fileName .. '_'

trainLabels = {}
testLabels = {}
trainFilesSet = {}
testFilesSet = {}

nTrain = 0
nTest = 0

for k1, v1 in pairs(datasets) do
    local labelsPath = dataDir .. v1 .. '/labels.txt'
    local file = assert(io.open(labelsPath, 'r'))

    local nLines = 0
    for line in file:lines() do
        nLines = nLines + 1
    end
    nTest = math.floor(nLines*ratioTest)
    nTrain = nLines - nTest

    local i = 0
    file:seek('set')
    for line in file:lines() do
        if i < nTrain then
            local label = tonumber(line)
            table.insert(trainLabels, label)
            for k2, v2 in pairs(imageTypes) do
                local fileName = dataDir .. v1 .. '/' .. v2 .. '/' .. v2 .. '-' .. i*skip .. '.jpg'
                if trainFilesSet[k2] == nil then
                    trainFilesSet[k2] = {}
                end
                table.insert(trainFilesSet[k2], fileName)
                --print(fileName .. ': ' .. label)
            end
        else
            local label = tonumber(line)
            table.insert(testLabels, label)
            for k2, v2 in pairs(imageTypes) do
                local fileName = dataDir .. v1 .. '/' .. v2 .. '/' .. v2 .. '-' .. i*skip .. '.jpg'
                if testFilesSet[k2] == nil then
                    testFilesSet[k2] = {}
                end
                table.insert(testFilesSet[k2], fileName)
                --print(fileName .. ': ' .. label)
            end
        end
        i = i+1
    end
end

-- in train, balance pos and neg and randomize the order
posIdx = torch.Tensor(trainLabels):nonzero()
negIdx = torch.add(torch.Tensor(trainLabels), -1):nonzero()
print('train set now: ', posIdx:size(1), negIdx:size(1))
if setRatioPosTrain then 
    local nPos = posIdx:size(1)
    local nNeg = math.floor(nPos*(1 - ratioPosTrain)/ratioPosTrain)
    if nNeg < negIdx:size(1) then 
        print('train set after: ', nPos, nNeg)
        nTrain = nPos + nNeg

        local trainLabelsTemp = {}
        local trainFilesSetTemp = {}

        for i = 1, nPos do
            local idx = posIdx[i][1]
            assert(trainLabels[idx] == 1)
            table.insert(trainLabelsTemp, trainLabels[idx])
            for k, v in pairs(imageTypes) do
                if trainFilesSetTemp[k] == nil then
                    trainFilesSetTemp[k] = {}
                end
                table.insert(trainFilesSetTemp[k], trainFilesSet[k][idx])
            end
        end

        math.randomseed(seed) -- not randomized

        for i = 1, nNeg do
            local rand = math.random(negIdx:size(1))
            while negIdx[rand][1] == -1 do
                rand = math.random(negIdx:size(1))
            end

            local idx = negIdx[rand][1]
            assert(trainLabels[idx] == 0)
            table.insert(trainLabelsTemp, trainLabels[idx])
            for k, v in pairs(imageTypes) do
                table.insert(trainFilesSetTemp[k], trainFilesSet[k][idx])
            end
            negIdx[rand] = -1
        end

        trainLabels = trainLabelsTemp
        trainFilesSet = trainFilesSetTemp

        -- randomize the order
        for i = 1, #trainLabels*2 do
            local idx1 = math.random(#trainLabels)
            local idx2 = math.random(#trainLabels)
            trainLabels[idx1], trainLabels[idx2] = trainLabels[idx2], trainLabels[idx1] 
            for k, v in pairs(imageTypes) do
                trainFilesSet[k][idx1], trainFilesSet[k][idx2] = trainFilesSet[k][idx2], trainFilesSet[k][idx1]
            end
        end
    end
end
-- table to tensor
trainLabels = torch.Tensor(trainLabels)
print('train labels: ' .. trainLabels:size(1))

-- in test, balance pos and neg and randomize the order
posIdx = torch.Tensor(testLabels):nonzero()
negIdx = torch.add(torch.Tensor(testLabels), -1):nonzero()
print('test set now: ', posIdx:size(1), negIdx:size(1))
if setRatioPosTest then 
    local nPos = posIdx:size(1)
    local nNeg = math.floor(nPos*(1 - ratioPosTest)/ratioPosTest)
    if nNeg < negIdx:size(1) then 
        print('test set after: ', nPos, nNeg)
        nTest = nPos + nNeg

        local testLabelsTemp = {}
        local testFilesSetTemp = {}

        for i = 1, nPos do
            local idx = posIdx[i][1]
            assert(testLabels[idx] == 1)
            table.insert(testLabelsTemp, testLabels[idx])
            for k, v in pairs(imageTypes) do
                if testFilesSetTemp[k] == nil then
                    testFilesSetTemp[k] = {}
                end
                table.insert(testFilesSetTemp[k], testFilesSet[k][idx])
            end
        end

        math.randomseed(seed) -- not randomized

        for i = 1, nNeg do
            local rand = math.random(negIdx:size(1))
            while negIdx[rand][1] == -1 do
                rand = math.random(negIdx:size(1))
            end

            local idx = negIdx[rand][1]
            assert(testLabels[idx] == 0)
            table.insert(testLabelsTemp, testLabels[idx])
            for k, v in pairs(imageTypes) do
                table.insert(testFilesSetTemp[k], testFilesSet[k][idx])
            end
            negIdx[rand] = -1
        end

        testLabels = testLabelsTemp
        testFilesSet = testFilesSetTemp
    end
end
-- table to tensor
testLabels = torch.Tensor(testLabels)
print('test labels: ' .. testLabels:size(1))

-- write path to all test images 
testPath = assert(io.open(testPathFile .. 'test_path.txt', 'w'))
for i = 1, nTest do 
    testPath:write(testFilesSet[1][i], '\n')
end
testPath:close()

-- read images
trainImagesSet = {}
testImagesSet = {}

for k1, v1 in pairs(trainFilesSet) do
    if trainImagesSet[k1] == nil then
        trainImagesSet[k1] = {}
    end
    for k2, v2 in pairs(v1) do
        if k2 % 100 == 0 then
            print('reading train data ' .. k1 .. '.' .. k2)
        end
        table.insert(trainImagesSet[k1], image.load(v2))
    end
end
for k1, v1 in pairs(testFilesSet) do
    if testImagesSet[k1] == nil then
        testImagesSet[k1] = {}
    end
    for k2, v2 in pairs(v1) do
        if k2 % 100 == 0 then
            print('reading test data ' .. k1 .. '.' .. k2)
        end
        table.insert(testImagesSet[k1], image.load(v2))
    end
end

imageSize = trainImagesSet[1][1]:size()
print(imageSize)
--print(trainImagesSet)
--print(testImagesSet)
--print(#labels)

nChannelsSet = {}
totalChannels = 0
for k, v in pairs(imageTypes) do
    nChannelsSet[k] = trainImagesSet[k][1]:size(1)
    totalChannels = totalChannels + nChannelsSet[k]
end
--print(nChannelsSet)

-- create and save data object
hh = {}
hh.train = {}
hh.test = {}

if coor[1]-boxSz/2 < 1 then
    x1 = 1
    x2 = boxSz
elseif coor[1]+boxSz/2-1 > imageSize[3] then
    x2 = imageSize[3]
    x1 = x2-boxSz+1
else
    x1 = coor[1]-boxSz/2
    x2 = coor[1]+boxSz/2-1 
end
if coor[2]-boxSz/2 < 1 then
    y1 = 1
    y2 = boxSz
elseif coor[2]+boxSz/2-1 > imageSize[2] then
    y2 = imageSize[2]
    y1 = y2-boxSz+1
else
    y1 = coor[2]-boxSz/2
    y2 = coor[2]+boxSz/2-1 
end

if crop then
    print('tl: ' .. x1 .. ',' .. y1, 'br: ' .. x2 .. ',' .. y2)
end

if crop then
    hh.train.data = torch.Tensor(trainLabels:size(1), totalChannels, boxSz, boxSz)
    hh.test.data = torch.Tensor(testLabels:size(1), totalChannels, boxSz, boxSz)
else
    hh.train.data = torch.Tensor(trainLabels:size(1), totalChannels, imageSize[2], imageSize[3])
    hh.test.data = torch.Tensor(testLabels:size(1), totalChannels, imageSize[2], imageSize[3])
end
for i = 1, trainLabels:size(1) do
    if i % 1000 == 0 then
        print('processing train data ' .. i)
    end
    local curIdx = 1
    for k, v in pairs(imageTypes) do
        if crop then
            hh.train.data[i][{{curIdx, curIdx+nChannelsSet[k]-1}, {}, {}}] = trainImagesSet[k][i][{{}, {y1, y2}, {x1, x2}}]
        else
            hh.train.data[i][{{curIdx, curIdx+nChannelsSet[k]-1}, {}, {}}] = trainImagesSet[k][i]
        end
        curIdx = curIdx + nChannelsSet[k]
    end
end
for i = 1, testLabels:size(1) do
    if i % 1000 == 0 then
        print('processing test data ' .. i)
    end
    local curIdx = 1
    for k, v in pairs(imageTypes) do
        if crop then
            hh.test.data[i][{{curIdx, curIdx+nChannelsSet[k]-1}, {}, {}}] = testImagesSet[k][i][{{}, {y1, y2}, {x1, x2}}]
        else
            hh.test.data[i][{{curIdx, curIdx+nChannelsSet[k]-1}, {}, {}}] = testImagesSet[k][i]
        end
        curIdx = curIdx + nChannelsSet[k]
    end
end

hh.train.labels = trainLabels -- tensor
hh.test.labels = testLabels -- tensor

print('saving file: ', dataFile)
torch.save(dataFile, hh)
