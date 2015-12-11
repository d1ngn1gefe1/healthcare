require 'cutorch'
require 'nn'
require 'cunn'
require 'optim'
    
-- parameters
dataFile = '/scail/scratch/group/vision/hospital/data/' 
resultsFile = '/scail/scratch/group/vision/hospital/src/ap/'
fileName = 'rgb_crop_19_21'
maxIter = 20

-- load data file and labels file
dataFile = dataFile .. fileName .. '.t7'
resultsFile = resultsFile .. fileName .. '_'

hh = torch.load(dataFile)

k = 10 -- k-fold cross-validation
sz = hh.train.data:size()
nChannels = sz[1]
width = sz[2]
height = sz[3]
print('nChannels: ' .. nChannels)
print('height: ' .. width)
print('width: ' .. height)

for t = 1, 1 do
    print('<----------' .. t .. '---------->')
    trainSet = {}
    testSet = {}

    trainSet.data = hh.train.data
    trainSet.labels = hh.train.labels
    nTrain = trainSet.labels:size(1)
    print(trainSet)
    for i = 1, nTrain do
      trainSet.labels[i] = trainSet.labels[i] + 1
    end

    testSet.data = hh.test.data
    testSet.labels = hh.test.labels
    nTest = testSet.labels:size(1) 
    print(testSet)

    pos = trainSet.labels:nonzero()
    neg = torch.add(trainSet.labels, -1):nonzero()
    print('train set: ', pos:size(1), neg:size(1), nTrain)

    pos = testSet.labels:nonzero()
    neg = torch.add(testSet.labels, -1):nonzero()
    print('test set: ', pos:size(1), neg:size(1), nTest)

    setmetatable(trainSet, 
        {__index = function(t, i) 
                        return {t.data[i], t.labels[i]} 
                    end}
    );
    trainSet.data = trainSet.data:double() -- convert the data from a ByteTensor to a DoubleTensor.
    
    function trainSet:size() 
        return self.data:size(1) 
    end
    
    setmetatable(testSet, 
        {__index = function(t, i) 
                        return {t.data[i], t.labels[i]} 
                    end}
    );
    testSet.data = testSet.data:double() -- convert the data from a ByteTensor to a DoubleTensor.
    
    function testSet:size() 
        return self.data:size(1) 
    end
    
    -- volume count
    print('volume count')
   
    sz = trainSet.data[1][1]:size()
    trainSet.data:resize(trainSet.data:size(1), sz[1]*sz[2])
    testSet.data:resize(testSet.data:size(1), sz[1]*sz[2])

    print(trainSet.data:size())
    print(trainSet.labels:size())
    print(testSet.data:size())
    print(testSet.labels:size())

    -- model
    pixVal = 35
    num = 500
    trainSet.labels2 = torch.Tensor(nTrain)
    for i = 1, nTrain do
        local count = 0
        for j = 1, sz[1]*sz[1] do
            if trainSet.data[i][j] > pixVal then
                count = count + 1
            end
        end
        if count > num then
            trainSet.labels2[i] = 1
        else
            trainSet.label2[i] = 0
        end
    end


--[[
    -- test accuracy
    testSet.data = testSet.data:double()   -- convert from Byte tensor to Double tensor
    
    correct = 0
    testProb = {}
    testScores = io.open(resultsFile .. 'test_scores.txt', 'w')
    testTrue = io.open(resultsFile .. 'test_true.txt', 'w') 
    for i = 1, nTest do 
        local groundtruth = testSet.labels[i] + 1
        local prediction = model:forward(testSet.data[i])
        local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
        if groundtruth == indices[1] then
            correct = correct + 1
        end
        testProb[i] = (1/prediction[2])/(1/prediction[1]+1/prediction[2])
        testScores:write(testProb[i], '\n')
        testTrue:write(testSet.labels[i], '\n')
    end
    print('test accuracy: ' .. correct .. '/' .. nTest, 100*correct/nTest .. ' %')
    testScores:close()
    testTrue:close()   
    
    -- test accuracy by class
    classPreds = {0, 0}
    classPerformance = {0, 0}
    classCounts = {0, 0}
    for i = 1, nTest do
        local groundtruth = testSet.labels[i] + 1
        classCounts[groundtruth] = classCounts[groundtruth] + 1
        local prediction = model:forward(testSet.data[i])
        local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
        classPreds[indices[1] ] = classPreds[indices[1] ] + 1
        if groundtruth == indices[1] then
            classPerformance[groundtruth] = classPerformance[groundtruth] + 1
        end
    end

    for i = 1, 2 do
        print('test accuracy of class ' .. (i-1) .. ': ' .. classPerformance[i] .. '/' .. classCounts[i], ((classCounts[i] == 0) and 100 or 100*classPerformance[i]/classCounts[i]) .. ' %')
    end

    --print(classPreds)
    --print(classPerformance)
    --print(classCounts)
--]]
    -- train accuracy
    correct = 0
    trainProb = {}
    trainScores = io.open(resultsFile .. 'train_scores.txt', 'w')
    trainTrue = io.open(resultsFile .. 'train_true.txt', 'w') 
    for i = 1, nTrain do
        local groundtruth = trainSet.labels[i] - 1
        if groundtruth == trainSet.labels2[i] then
            correct = correct + 1
        end
        trainScores:write(trainSet.labels2[i], '\n')
        trainTrue:write(trainSet.labels[i]-1, '\n')
    end
    print('train accuracy: ' .. correct .. '/' .. nTrain, 100*correct/nTrain .. ' %')
    trainScores:close()
    trainTrue:close()   
--[[ 
    -- train accuracy by class
    classPreds = {0, 0}
    classPerformance = {0, 0}
    classCounts = {0, 0}
    for i = 1, trainSet:size() do
        local groundtruth = trainSet.labels[i]
        classCounts[groundtruth] = classCounts[groundtruth] + 1
        local prediction = model:forward(trainSet.data[i])
        local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
        classPreds[indices[1] ] = classPreds[indices[1] ] + 1
        if groundtruth == indices[1] then
            classPerformance[groundtruth] = classPerformance[groundtruth] + 1
        end
    end
    
    for i = 1, 2 do
        print('train accuracy of class ' .. (i-1) .. ': ' .. classPerformance[i] .. '/' .. classCounts[i], ((classCounts[i] == 0) and 100 or 100*classPerformance[i]/classCounts[i]) .. ' %')
    end

    --print(class_preds)
    --print(class_performance)
    --print(class_counts)    
--]]
end
