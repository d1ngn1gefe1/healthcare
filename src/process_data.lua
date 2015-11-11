require 'cutorch'
require 'nn'
require 'image'

-- READ TRAIN IMAGES

imageTypes = {'rgb'}
dataDir = '/scail/data/group/vision/u/syyeung/hospital/data/'
datasets = {'cvpr10-19-15morning/'}

dir = '/scail/scratch/group/vision/hospital/'
coor = {110, 240}
boxSz = 64
crop = true
random = false

testPathFile = dir .. 'src/ap/'
for i = 1, #imageTypes do
    testPathFile = testPathFile .. imageTypes[i] .. '_'
end
if crop then
    testPathFile = testPathFile .. 'crop_'
end
if random then
    testPathFile = testPathFile .. 'random_'
end

skip = 3
ratioTest = 0.2
setRatioPosTrain = true
setRatioPosTest = true
ratioPosTrain = 0.1
ratioPosTest = 0.1

labels = {}
filesSet = {}
for k, v in pairs(imageTypes) do
    filesSet[k] = {}
end

for k1, v1 in pairs(datasets) do
    local labelsDir = dataDir .. v1 .. 'labels.txt'
    file = io.open(labelsDir)
    if file then
        local i = 0
        for line in file:lines() do
            local label = tonumber(line)
            table.insert(labels, label)
            for k2, v2 in pairs(imageTypes) do
                local fileName = dataDir .. v1 .. v2 .. '/' .. v2 .. '-' .. i*skip .. '.jpg'
                table.insert(filesSet[k2], fileName)
                --print(fileName .. ': ' .. label)
            end
            i = i+1
        end
    else
        print('label file does not exist')
    end
end

-- will contaminate the test set
if random then
    for i = 1, #labels*2 do
        local idx1 = math.random(#labels)
        local idx2 = math.random(#labels)
        labels[idx1], labels[idx2] = labels[idx2], labels[idx1] 
        for k, v in pairs(imageTypes) do
            filesSet[k][idx1], filesSet[k][idx2] = filesSet[k][idx2], filesSet[k][idx1]
        end
    end
end
--

labelsTensor = torch.Tensor(#labels)
for i = 1, #labels do
    labelsTensor[i] = labels[i]
end

-- split train and test
nTest = math.floor(#labels*ratioTest)
nTrain = #labels - nTest
trainLabels = labelsTensor:narrow(1, nTest + 1, nTrain)
testLabels = labelsTensor:narrow(1, 1, nTest)
print(trainLabels:size(), testLabels:size())
trainFilesSet = {}
testFilesSet = {}
for k, v in pairs(imageTypes) do
    trainFilesSet[k] = {}
    testFilesSet[k] = {}
    for i = 1, #filesSet[k] do
        if i <= nTest then
            table.insert(testFilesSet[k], filesSet[k][i])
        else
            table.insert(trainFilesSet[k], filesSet[k][i])
        end
    end
end

-- in train, balance pos and neg and randomize the order
if setRatioPosTrain then 
    pos = trainLabels:nonzero()
    neg = torch.add(trainLabels, -1):nonzero()
    print('train set now: ', pos:size(1), neg:size(1))
    nPos = pos:size(1)
    nNeg = math.floor(nPos*(1 - ratioPosTrain)/ratioPosTrain)
    if nNeg < neg:size(1) then 
        print('train set after: ', nPos, nNeg)
        nTrain = nPos + nNeg

        trainLabels2 = {}
        trainFilesSet2 = {}
        for k, v in pairs(imageTypes) do
            trainFilesSet2[k] = {}
        end

        for i = 1, nPos do
            local idx = pos[i][1]
            assert(trainLabels[idx] == 1)
            table.insert(trainLabels2, trainLabels[idx])
            for k, v in pairs(imageTypes) do
                table.insert(trainFilesSet2[k], trainFilesSet[k][idx])
            end
        end

        math.randomseed(1) -- not randomized

        for i = 1, nNeg do
            local rand = math.random(neg:size(1))
            while neg[rand][1] == -1 do
                rand = math.random(neg:size(1))
            end

            local idx = neg[rand][1]
            assert(trainLabels[idx] == 0)
            table.insert(trainLabels2, trainLabels[idx])
            for k, v in pairs(imageTypes) do
                table.insert(trainFilesSet2[k], trainFilesSet[k][idx])
            end
            neg[rand] = -1
        end

        --labelsTensor = torch.Tensor(#labels2)
        --for i = 1, #labels2 do
        --    labelsTensor[i] = labels2[i]
        --end
        --pos = labelsTensor:nonzero()
        --neg = torch.add(labelsTensor, -1):nonzero()
        --print(pos:size(1), neg:size(1))

        trainLabels = trainLabels2
        trainFilesSet = trainFilesSet2

        for i = 1, #trainLabels*2 do
            local idx1 = math.random(#trainLabels)
            local idx2 = math.random(#trainLabels)
            trainLabels[idx1], trainLabels[idx2] = trainLabels[idx2], trainLabels[idx1] 
            for k, v in pairs(imageTypes) do
                trainFilesSet[k][idx1], trainFilesSet[k][idx2] = trainFilesSet[k][idx2], trainFilesSet[k][idx1]
            end
        end

        -- array to tensor
        tmp = torch.Tensor(#trainLabels)
        for i = 1, #trainLabels do
            tmp[i] = trainLabels[i]
        end
        trainLabels = tmp
    end
end

-- in test, balance pos and neg and randomize the order
if setRatioPosTest then 
    pos = testLabels:nonzero()
    neg = torch.add(testLabels, -1):nonzero()
    print('test set now: ', pos:size(1), neg:size(1))
    nPos = pos:size(1)
    nNeg = math.floor(nPos*(1 - ratioPosTest)/ratioPosTest)
    if nNeg < neg:size(1) then 
        print('test set after: ', nPos, nNeg)
        nTest = nPos + nNeg

        testLabels2 = {}
        testFilesSet2 = {}
        for k, v in pairs(imageTypes) do
            testFilesSet2[k] = {}
        end

        for i = 1, nPos do
            local idx = pos[i][1]
            assert(testLabels[idx] == 1)
            table.insert(testLabels2, testLabels[idx])
            for k, v in pairs(imageTypes) do
                table.insert(testFilesSet2[k], testFilesSet[k][idx])
            end
        end

        math.randomseed(1) -- not randomized

        for i = 1, nNeg do
            local rand = math.random(neg:size(1))
            while neg[rand][1] == -1 do
                rand = math.random(neg:size(1))
            end

            local idx = neg[rand][1]
            assert(testLabels[idx] == 0)
            table.insert(testLabels2, testLabels[idx])
            for k, v in pairs(imageTypes) do
                table.insert(testFilesSet2[k], testFilesSet[k][idx])
            end
            neg[rand] = -1
        end

        --labelsTensor = torch.Tensor(#labels2)
        --for i = 1, #labels2 do
        --    labelsTensor[i] = labels2[i]
        --end
        --pos = labelsTensor:nonzero()
        --neg = torch.add(labelsTensor, -1):nonzero()
        --print(pos:size(1), neg:size(1))

        testLabels = testLabels2
        testFilesSet = testFilesSet2

        for i = 1, #testLabels*2 do
            local idx1 = math.random(#testLabels)
            local idx2 = math.random(#testLabels)
            testLabels[idx1], testLabels[idx2] = testLabels[idx2], testLabels[idx1] 
            for k, v in pairs(imageTypes) do
                testFilesSet[k][idx1], testFilesSet[k][idx2] = testFilesSet[k][idx2], testFilesSet[k][idx1]
            end
        end

        -- array to tensor
        tmp = torch.Tensor(#testLabels)
        for i = 1, #testLabels do
            tmp[i] = testLabels[i]
        end
        testLabels = tmp
    end
end

testPath = io.open(testPathFile .. 'test_path.txt', 'w')
for i = 1, nTest do 
    testPath:write(testFilesSet[1][i], '\n')
end
testPath:close()

-- read images
trainImagesSet = {}
testImagesSet = {}
for k, v in pairs(imageTypes) do
    trainImagesSet[k] = {}
    testImagesSet[k] = {}
end

for k1, v1 in pairs(trainFilesSet) do
    for k2, v2 in pairs(v1) do
        table.insert(trainImagesSet[k1], image.load(v2))
    end
end
for k1, v1 in pairs(testFilesSet) do
    for k2, v2 in pairs(v1) do
        table.insert(testImagesSet[k1], image.load(v2))
    end
end

imageSize = trainImagesSet[1][1]:size()
print(imageSize)
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
    curIdx = 1
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
    curIdx = 1
    for k, v in pairs(imageTypes) do
        if crop then
            hh.test.data[i][{{curIdx, curIdx+nChannelsSet[k]-1}, {}, {}}] = testImagesSet[k][i][{{}, {y1, y2}, {x1, x2}}]
        else
            hh.test.data[i][{{curIdx, curIdx+nChannelsSet[k]-1}, {}, {}}] = testImagesSet[k][i]
        end
        curIdx = curIdx + nChannelsSet[k]
    end
end

hh.train.labels = trainLabels 
hh.test.labels = testLabels

dataFile = '../data/hh'
for i = 1, #imageTypes do
    dataFile = dataFile .. '_' .. imageTypes[i]
end
if crop then
    dataFile = dataFile .. '_crop'
end
if random then
    dataFile = dataFile .. '_random'
end
dataFile = dataFile .. '.t7'
print('saving file: ', dataFile)
torch.save(dataFile, hh)
