require 'cutorch'
require 'nn'
require 'cunn'
require 'optim'
    
-- parameters
imageTypes = {'d'}
dir = '/scail/scratch/group/vision/hospital/'
crop = true
maxIter = 20
volume = false

-- load data file and labels file
dataFile = dir .. 'data/hh' 
labelsFile = dir .. 'src/ap/'
for i = 1, #imageTypes do
    dataFile = dataFile .. '_' .. imageTypes[i]
    labelsFile = labelsFile .. imageTypes[i] .. '_'
end
if crop then
    dataFile = dataFile .. '_crop.t7'
    labelsFile = labelsFile .. 'crop_baseline_'
else
    dataFile = dataFile .. '.t7'
end

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
    
    -- logistic regression
    print('logistic regression')
    sz = trainSet.data[1][1]:size()
    trainSet.data:resize(trainSet.data:size(1), sz[1]*sz[2])
    testSet.data:resize(testSet.data:size(1), sz[1]*sz[2])
    
    if volume == true then
    trainData = torch.Tensor(trainSet.data:size(1))
    for i = 1,trainSet.data:size(1) do
      local sum = 0
      for j = 1,trainSet.data:size(2) do
        sum = sum + trainSet.data[i][j]
      end
      trainData[i] = sum
    end
    trainSet.data = trainData
    trainSet.data = trainSet.data:double()
    end

    if volume == true then
      testData = torch.Tensor(testSet.data:size(1))
      for i = 1,testSet.data:size(1) do
        local sum = 0
	for j = 1,testSet.data:size(2) do
          sum = sum + testSet.data[i][j]
        end
	testData[i] = sum
      end
      testSet.data = testData
      testSet.data = testSet.data:double()
    end
    print(trainSet.data:size())
    print(testSet.data:size())

    model = nn.Sequential()
    if volume == true then
      model:add(nn.Linear(trainSet.data:size(1), 2))
    else
      model:add(nn.Linear(sz[1]*sz[2], 2))
    end
    model:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()

    x, dl_dx = model:getParameters()
    
    feval = function(x_new)
        if x ~= x_new then
            x:copy(x_new)
        end
     
        _nidx_ = (_nidx_ or 0) + 1
        if _nidx_ > trainSet.data:size(1) then _nidx_ = 1 end

        local inputs = trainSet.data[_nidx_]
        local target = trainSet.labels[_nidx_]

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
    testScores = io.open(labelsFile .. 'test_scores.txt', 'w')
    testTrue = io.open(labelsFile .. 'test_true.txt', 'w') 
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

    -- train accuracy
    correct = 0
    trainProb = {}
    trainScores = io.open(labelsFile .. 'train_scores.txt', 'w')
    trainTrue = io.open(labelsFile .. 'train_true.txt', 'w') 
    for i = 1, nTrain do
        local groundtruth = trainSet.labels[i]
        local prediction = model:forward(trainSet.data[i])
        local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
        if groundtruth == indices[1] then
            correct = correct + 1
        end
        trainProb[i] = (1/prediction[2])/(1/prediction[1]+1/prediction[2])
        trainScores:write(trainProb[i], '\n')
        trainTrue:write(trainSet.labels[i]-1, '\n')
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
end
