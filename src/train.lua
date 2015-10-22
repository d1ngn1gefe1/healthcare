require 'cutorch'
require 'nn'
require 'cunn'
    
-- load data file
imageTypes = {'d', 'rgb'}
dir = '/scail/scratch/group/vision/hospital/data/'
crop = true
dataFile = dir .. 'hh' 
for i = 1, #imageTypes do
    dataFile = dataFile .. '_' .. imageTypes[i]
end
if crop then
    dataFile = dataFile .. '_crop'
end
dataFile = dataFile .. '.t7'
hh = torch.load(dataFile)

    
k = 10 -- k-fold cross-validation
sz = hh.images[1]:size()
nChannels = sz[1]
width = sz[2]
height = sz[3]
print('nChannels: ' .. nChannels)
print('height: ' .. width)
print('width: ' .. height)

trainAccur = 0
testAccur = 0
   
for t = 1, k do
    print('<----------' .. t .. '---------->')
    trainSet = {}
    testSet = {}
    nTotal = hh.labels:size(1);
    nTest = math.floor(nTotal/k)
    nTrain = nTotal-nTest
    
    print(nTotal .. ',' .. nTrain .. ',' .. nTest)
    
    testData = hh.images:narrow(1, (t-1)*nTest+1, nTest)
    testLabels = hh.labels:narrow(1, (t-1)*nTest+1, nTest)

    print(1, (t-1)*nTest, t*nTest+1, nTrain-(t-1)*nTest)

    if t == 1 then
        trainData = hh.images:narrow(1, nTest+1, nTrain)
        trainLabels = hh.labels:narrow(1, nTest+1, nTrain)
    elseif t == k then
        trainData = hh.images:narrow(1, 1, nTrain)
        trainLabels = hh.labels:narrow(1, 1, nTrain)
    else
        trainData = torch.cat(hh.images:narrow(1, 1, (t-1)*nTest), hh.images:narrow(1, t*nTest+1, nTrain-(t-1)*nTest), 1)
        trainLabels = torch.cat(hh.labels:narrow(1, 1, (t-1)*nTest), hh.labels:narrow(1, t*nTest+1, nTrain-(t-1)*nTest), 1)
    end

    trainPos = trainLabels:nonzero()
    trainNeg = torch.add(trainLabels, -1):nonzero()
    testPos = testLabels:nonzero()
    testNeg = torch.add(testLabels, -1):nonzero()
  
    print('train: pos ' .. ((trainPos:nDimension() == 0) and 0 or trainPos:size(1)) .. ', neg ' .. ((trainNeg:nDimension() == 0) and 0 or trainNeg:size(1)))
    print('test: pos ' .. ((testPos:nDimension() == 0) and 0 or testPos:size(1)) .. ', neg ' .. ((testNeg:nDimension() == 0) and 0 or testNeg:size(1)))
    --print(hh.labels)
 
    -- create training set with every other example a negative
    negStride = math.floor(trainNeg:size(1)/trainPos:size(1))
    nTrainPairs = trainPos:size(1)
    nTrain = nTrainPairs*2
    trainData2 = torch.Tensor(nTrain, nChannels, height, width)
    trainLabels2 = torch.Tensor(nTrain)
    for i = 1, nTrainPairs do
        local posIdx = trainPos[i]
        local negIdx = trainNeg[(i-1)*negStride+1]
    
        trainData2[2*i-1] = trainData:narrow(1, posIdx[1], 1):float()
        trainLabels2[2*i-1] = trainLabels:narrow(1, posIdx[1], 1) + 1
    
        trainData2[2*i] = trainData:narrow(1, negIdx[1], 1):float()
        trainLabels2[2*i] = trainLabels:narrow(1, negIdx[1], 1) + 1
    end
    
    trainSet.data = trainData2
    trainSet.labels = trainLabels2
    print(trainSet)
        
    testSet.data = testData
    testSet.labels = testLabels
    print(testSet)
    
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
    trainer.maxIteration = 20
    
    trainer:train(trainSet)
    
    -- test accuracy
    testSet.data = testSet.data:double()   -- convert from Byte tensor to Double tensor
    for i = 1, nChannels do -- over each image channel
        testSet.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
        testSet.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
    end
    testSet.data = testSet.data:cuda()
    
    correct = 0
    for i = 1, nTest do 
        local groundtruth = testSet.labels[i] + 1
        local prediction = net:forward(testSet.data[i])
        local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
        if groundtruth == indices[1] then
            correct = correct + 1
        end
    end
    testAccur = testAccur + correct/nTest
    print('test accuracy: ' .. correct .. '/' .. nTest, 100*correct/nTest .. ' %')
    
    -- test accuracy by class
    classPreds = {0, 0}
    classPerformance = {0, 0}
    classCounts = {0, 0}
    for i = 1, nTest do
        local groundtruth = testSet.labels[i] + 1
        classCounts[groundtruth] = classCounts[groundtruth] + 1
        local prediction = net:forward(testSet.data[i])
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
    
    -- train accuracy
    correct = 0
    for i = 1, nTrain do
        local groundtruth = trainSet.labels[i]
        local prediction = net:forward(trainSet.data[i])
        local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
        if groundtruth == indices[1] then
            correct = correct + 1
        end
    end
    trainAccur = trainAccur + correct/nTrain 
    print('train accuracy: ' .. correct .. '/' .. nTrain, 100*correct/nTrain .. ' %')
    
    -- train accuracy by class
    classPreds = {0, 0}
    classPerformance = {0, 0}
    classCounts = {0, 0}
    for i = 1, trainSet:size() do
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

print('total test accuracy: ', testAccur/k)
print('total train accuracy: ', trainAccur/k)    
