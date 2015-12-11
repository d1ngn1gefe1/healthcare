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
    --for i = 1, nTrain do
    --  trainSet.labels[i] = trainSet.labels[i] + 1
    --  print(trainSet.labels[i])
    --end

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

    -- labels convert to svm format
    for i = 1, nTrain do
      trainSet.labels[i] = trainSet.labels[i]*2 - 1
      --print(trainSet.labels[i])
    end
    for i = 1, nTest do
      testSet.labels[i] = testSet.labels[i]*2 - 1
      --print(testSet.labels[i])
    end

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
    
    -- svm
    print('svm')

    sz = trainSet.data[1][1]:size()
    trainSet.data:resize(trainSet.data:size(1), sz[1]*sz[2])
    testSet.data:resize(testSet.data:size(1), sz[1]*sz[2])

    print(trainSet.data:size())
    print(trainSet.labels:size())
    print(testSet.data:size())
    print(testSet.labels:size())

    -- model
    model = nn.Sequential()
    model:add(nn.Linear(sz[1]*sz[2], 1))
    criterion = nn.MarginCriterion()

    x, dl_dx = model:getParameters()
    
    feval = function(x_new)
        if x ~= x_new then
            x:copy(x_new)
        end
     
        _nidx_ = (_nidx_ or 0) + 1
        if _nidx_ > trainSet.data:size(1) then _nidx_ = 1 end

        local inputs = trainSet.data[_nidx_]

        local target = torch.Tensor{trainSet.labels[_nidx_]}

        dl_dx:zero()
     
        local loss_x = criterion:forward(model:forward(inputs), target)
        model:backward(inputs, criterion:backward(model.output, target))
 
        return loss_x, dl_dx
    end

    sgd_params = {
        learningRate = 1e-2,
        learningRateDecay = 1e-4,
        weightDecay = 0,
        momentum = 0
    }

    epochs = 100
    
    print('Training with SGD')

    for i = 1, epochs do
        current_loss = 0
        for i = 1, trainSet.data:size(1) do
            _, fs = optim.sgd(feval, x, sgd_params)
            current_loss = current_loss + fs[1]
        end
 
        current_loss = current_loss / trainSet.data:size(1)
        print('epoch = ' .. i .. ' of ' .. epochs .. ' current loss = ' .. current_loss)
    end

    -- test accuracy
    testSet.data = testSet.data:double()   -- convert from Byte tensor to Double tensor
    
    correct = 0
    testProb = {}
    testScores = io.open(resultsFile .. 'test_scores.txt', 'w')
    testTrue = io.open(resultsFile .. 'test_true.txt', 'w') 
    for i = 1, nTest do 
        local groundtruth = testSet.labels[i]
        local prediction = model:forward(testSet.data[i])
        if groundtruth*prediction[1] > 0 then
            correct = correct + 1
        end
        testProb[i] = prediction[1]
        testScores:write(testProb[i], '\n')
        testTrue:write((testSet.labels[i]+1)/2, '\n')
    end
    print('test accuracy: ' .. correct .. '/' .. nTest, 100*correct/nTest .. ' %')
    testScores:close()
    testTrue:close()   

    -- test accuracy by class
    classPreds = {0, 0}
    classPerformance = {0, 0}
    classCounts = {0, 0}
    for i = 1, nTest do
        local groundtruth = testSet.labels[i]
        local idx = (groundtruth+1)/2+1
        classCounts[idx] = classCounts[idx] + 1
        local prediction = model:forward(testSet.data[i])
        if groundtruth*prediction[1] > 0 then
            classPerformance[idx] = classPerformance[idx] + 1
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
    trainProb = {}
    trainScores = io.open(resultsFile .. 'train_scores.txt', 'w')
    trainTrue = io.open(resultsFile .. 'train_true.txt', 'w') 
    for i = 1, nTrain do
        local groundtruth = trainSet.labels[i]
        local prediction = model:forward(trainSet.data[i])
        if groundtruth*prediction[1] > 0 then
            correct = correct + 1
        end
        trainProb[i] = prediction[1]
        trainScores:write(trainProb[i], '\n')
        trainTrue:write((trainSet.labels[i]+1)/2, '\n')
    end
    print('train accuracy: ' .. correct .. '/' .. nTrain, 100*correct/nTrain .. ' %')
    trainScores:close()
    trainTrue:close()   

    -- train accuracy by class
    classPreds = {0, 0}
    classPerformance = {0, 0}
    classCounts = {0, 0}
    for i = 1, trainSet:size() do
        local groundtruth = trainSet.labels[i]
        local idx = (groundtruth+1)/2+1
        classCounts[idx] = classCounts[idx] + 1
        local prediction = model:forward(trainSet.data[i])
        if groundtruth*prediction[1] > 0 then
            classPerformance[idx] = classPerformance[idx] + 1
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
