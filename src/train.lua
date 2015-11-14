require 'cutorch'
require 'nn'
require 'cunn'
    
-- parameters
crop = true
maxIter = 10
k = 1 -- k-fold cross-validation
fileName = 'rgb_crop_19_21'
kernel = {0.1, 0.15, 0.15, 0.2, 0.15, 0.15, 0.1}
smooth = true
-- end parameters

-- load data file and labels file
dataFile = '/scail/scratch/group/vision/hospital/data/' 
resultsFile = '/scail/scratch/group/vision/hospital/src/ap/'
dataFile = dataFile .. fileName .. '.t7'
resultsFile = resultsFile .. fileName .. '_'

hh = torch.load(dataFile)

sz = hh.train.data[1]:size()
nChannels = sz[1]
width = sz[2]
height = sz[3]
print('nChannels: ' .. nChannels)
print('height: ' .. width)
print('width: ' .. height)

for t = 1, k do
    print('<----------' .. t .. '---------->')
    trainSet = {}
    testSet = {}
    
    trainSet.data = hh.train.data
    trainSet.labels = hh.train.labels
    nTrain = trainSet.labels:size(1)
    print(trainSet)
        
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

    trainSet.labels = trainSet.labels + 1

    -- normalization
    mean = {} -- store the mean, to normalize the test set in the future
    stdv  = {} -- store the standard-deviation for the future
    for i = 1, nChannels do -- over each image channel
        mean[i] = trainSet.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
        print('Channel ' .. i .. ', Mean: ' .. mean[i])
        trainSet.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
        stdv[i] = trainSet.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
        print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
        trainSet.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
    end
    
    -- view an image
    -- itorch.image(trainset.data[301])
    
    -- define network
    net = nn.Sequential() -- input: nChannels, 64, 64
    if crop == true then
        net:add(nn.SpatialConvolution(nChannels, 6, 5, 5)) -- output: 6, 60, 60
        net:add(nn.SpatialMaxPooling(3, 3, 3, 3)) -- output: 6, 20, 20
        net:add(nn.SpatialConvolution(6, 12, 5, 5)) -- output: 12, 16, 16
        net:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- output: 12, 8, 8

        net:add(nn.View(12*8*8))
        net:add(nn.Linear(12*8*8, 100))
        net:add(nn.Linear(100, 2))
        net:add(nn.LogSoftMax())
    else 
        net:add(nn.SpatialConvolution(nChannels, 6, 7, 7, 2, 2))
        net:add(nn.SpatialMaxPooling(3, 3, 3, 3))  
        net:add(nn.SpatialConvolution(6, 16, 5, 5, 2, 2))
        net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
        
        net:add(nn.View(16*9*12))
        net:add(nn.Linear(16*9*12, 50))
        net:add(nn.Linear(50, 2))
        net:add(nn.LogSoftMax())                    
    end 

    net = net:cuda()
    criterion = nn.ClassNLLCriterion()
    
    --train
    net = net:cuda()
    criterion = criterion:cuda()
    trainSet.data = trainSet.data:cuda()
    
    trainer = nn.StochasticGradient(net, criterion)
    trainer.learningRate = 0.001
    trainer.maxIteration = maxIter
    
    trainer:train(trainSet)
    
    -- test accuracy
    testSet.data = testSet.data:double()   -- convert from Byte tensor to Double tensor
    for i = 1, nChannels do -- over each image channel
        testSet.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
        testSet.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
    end
    testSet.data = testSet.data:cuda()
    
    testScores = io.open(resultsFile .. 'test_scores.txt', 'w')
    testTrue = io.open(resultsFile .. 'test_true.txt', 'w') 
    testPredict = io.open(resultsFile .. 'test_predict.txt', 'w')
    testDiff = io.open(resultsFile .. 'test_diff.txt', 'w')
    testProb = {}
    correct = 0
    truePos = 0
    trueNeg = 0
    nPos = 0
    nNeg = 0
    for i = 1, nTest do 
        local prediction = net:forward(testSet.data[i])
        if prediction[1] == 0 then
            testProb[i] = 0
        elseif prediction[2] == 0 then
            testProb[i] = 1
        else
            testProb[i] = (1/prediction[2])/(1/prediction[1] + 1/prediction[2])
        end
        local groundtruth = testSet.labels[i]
        nPos = nPos + groundtruth
        nNeg = nNeg - (groundtruth - 1)
        testTrue:write(groundtruth, '\n')
        if groundtruth == math.floor(testProb[i] + 0.5) then
            correct = correct + 1
            if groundtruth == 1 then
                truePos = truePos + 1
            else
                trueNeg = trueNeg + 1
            end
        end
        testScores:write(testProb[i], '\n')
        testPredict:write(math.floor(testProb[i] + 0.5), '\n')
        testDiff:write(groundtruth - math.floor(testProb[i] + 0.5), '\n')
    end
    testScores:close()
    testTrue:close()
    testPredict:close()
    testDiff:close()
    print('test accuracy: ' .. correct .. '/' .. nTest, 100*correct/nTest .. ' %')
    print('test accuracy of class 0: ' .. trueNeg .. '/' .. nNeg, (trueNeg/nNeg) .. ' %')
    print('test accuracy of class 1: ' .. truePos .. '/' .. nPos, (truePos/nPos) .. ' %')

    if smooth then
        testSmoothScores = io.open(resultsFile .. 'test_smooth_scores.txt', 'w')
        testSmoothPredict = io.open(resultsFile .. 'test_smooth_predict.txt', 'w')
        testSmoothDiff = io.open(resultsFile .. 'test_smooth_diff.txt', 'w')
        testProbSmooth = {}
        correct = 0
        truePos = 0
        trueNeg = 0
        nPos = 0
        nNeg = 0
        local r = math.floor(#kernel/2)
        for i = 1, nTest do 
            local groundtruth = testSet.labels[i]
            nPos = nPos + groundtruth
            nNeg = nNeg - (groundtruth - 1)
            testProbSmooth[i] = 0
            for j = i - r, i + r do
                if j >= 1 and j <= nTest then
                    testProbSmooth[i] = testProbSmooth[i] + kernel[j-(i-r)+1]*testProb[j]
                end
            end
            local predict = math.floor(testProbSmooth[i] + 0.5)
            if groundtruth == predict then
                correct = correct + 1
                if groundtruth == 1 then
                    truePos = truePos + 1
                else
                    trueNeg = trueNeg + 1
                end
            end
            testSmoothScores:write(testProbSmooth[i], '\n')
            testSmoothPredict:write(predict, '\n')
            testSmoothDiff:write(groundtruth - math.floor(testProbSmooth[i] + 0.5), '\n')
        end
        testSmoothScores:close()
        testSmoothPredict:close()
        testSmoothDiff:close()
        print('test accuracy smoothed: ' .. correct .. '/' .. nTest, 100*correct/nTest .. ' %')
        print('test accuracy smoothed of class 0: ' .. trueNeg .. '/' .. nNeg, (trueNeg/nNeg) .. ' %')
        print('test accuracy smoothed of class 1: ' .. truePos .. '/' .. nPos, (truePos/nPos) .. ' %')
    end
    
    -- train accuracy
    correct = 0
    trainScores = io.open(resultsFile .. 'train_scores.txt', 'w')
    trainTrue = io.open(resultsFile .. 'train_true.txt', 'w') 
    trainPredict = io.open(resultsFile .. 'train_predict.txt', 'w') 
    for i = 1, nTrain do
        local groundtruth = trainSet.labels[i]
        local prediction = net:forward(trainSet.data[i])
        local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
        if groundtruth == indices[1] then
            correct = correct + 1
        end
        local trainProb = 0
        if prediction[1] == 0 then
            trainProb = 0
        elseif prediction[2] == 0 then
            trainProb = 1
        else
            trainProb = (1/prediction[2])/(1/prediction[1] + 1/prediction[2])
        end
        trainScores:write(trainProb, '\n')
        trainTrue:write(trainSet.labels[i]-1, '\n')
        trainPredict:write(indices[1]-1, '\n')
    end
    print('train accuracy: ' .. correct .. '/' .. nTrain, 100*correct/nTrain .. ' %')
    trainScores:close()
    trainTrue:close()   
    trainPredict:close()
 
    -- train accuracy by class
    classPreds = {0, 0}
    classPerformance = {0, 0}
    classCounts = {0, 0}
    for i = 1, nTrain do
        local groundtruth = trainSet.labels[i]
        classCounts[groundtruth] = classCounts[groundtruth] + 1
        local prediction = net:forward(trainSet.data[i])
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
    
end
