require 'cutorch'
require 'nn'
require 'cunn'

numChannels = 3

-- load data file
srcDir = '/scail/scratch/group/vision/bypeng/healthcare/data/'
dataFile_pos = srcDir .. 'hh_positive.t7'
dataFile_neg = srcDir .. 'hh_negative.t7'

hh_pos = torch.load(dataFile_pos)
hh_neg = torch.load(dataFile_neg)

numPos = hh_pos.data:size(1) -- 11724
numNeg = hh_neg.data:size(1) -- 2436

trainNumPos = math.floor(numPos * 0.7)
trainNumNeg = math.floor(numNeg * 0.7)
trainNum = 0

if (trainNumPos > trainNumNeg) then
  trainNum = trainNumNeg
else
  trainNum = trainNumPos
end

testNumPos = numPos - trainNum
testNumNeg = numNeg - trainNum
testNum = 0

if (testNumPos > testNumNeg) then
  testNum = testNumNeg
else
  testNum = testNumPos
end

trainSet = {}
trainData = torch.Tensor(trainNum*2, 3, 32, 32)
trainLabels = torch.Tensor(trainNum*2)

for i = 1, trainNum do
  trainData[2*i-1] = hh_pos.data:narrow(1,i,1):float()
  trainLabels[2*i-1] = hh_pos.labels:narrow(1,i,1):float() + 1
  trainData[2*i] = hh_neg.data:narrow(1,i,1):float()
  trainLabels[2*i] = hh_neg.labels:narrow(1,i,1):float() + 1
end

trainSet.data = trainData
trainSet.labels = trainLabels

testSet = {}
testData = torch.Tensor(testNum*2, 3, 32, 32)
testLabels = torch.Tensor(testNumPos*2)

for i = 1, testNum do
  testData[2*i-1] = hh_pos.data:narrow(1,i+trainNum,1):float()
  testLabels[2*i-1] = hh_pos.labels:narrow(1,i+trainNum,1):float()
  testData[2*i] = hh_neg.data:narrow(1,i+trainNum,1):float()
  testLabels[2*i] = hh_neg.labels:narrow(1,i+trainNum,1):float()
end

testSet.data = testData
testSet.labels = testLabels

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

mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,numChannels do -- over each image channel
    mean[i] = trainSet.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainSet.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    stdv[i] = trainSet.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainSet.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- Define network
net = nn.Sequential() -- input: 3, 32, 32
net:add(nn.SpatialConvolution(numChannels, 6, 5, 5)) -- output: 6, 28, 28
net:add(nn.SpatialMaxPooling(2,2,2,2)) -- output: 6, 14, 14
net:add(nn.SpatialConvolution(6, 12, 5, 5)) -- output; 12, 10, 10
net:add(nn.SpatialMaxPooling(2,2,2,2)) -- output: 12, 5, 5

net:add(nn.View(12*5*5)) -- output: 300, 1
net:add(nn.Linear(300, 10))
net:add(nn.Linear(10, 2))
net:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()

-- Train
net = net:cuda()
criterion = criterion:cuda()
trainSet.data = trainSet.data:cuda()

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 10

trainer:train(trainSet)

-- Save model
torch.save('handClassifier.bin', net)

-- test accuracy
testSet.data = testSet.data:double()   -- convert from Byte tensor to Double tensor
for i=1,numChannels do -- over each image channel
    testSet.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testSet.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end
testSet.data = testSet.data:cuda()

testSize = #testSet.data
numTestExamples = testSize[1]
correct = 0
for i=1,numTestExamples do
    local groundtruth = testSet.labels[i] + 1
    local prediction = net:forward(testSet.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print(correct, 100*correct/numTestExamples .. ' % Test Accuracy')

-- test accuracy by class
class_preds = {0, 0}
class_performance = {0, 0}
class_counts = {0, 0}
for i=1,numTestExamples do
    local groundtruth = testSet.labels[i] + 1
    class_counts[groundtruth] = class_counts[groundtruth] + 1
    local prediction = net:forward(testSet.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    class_preds[indices[1]] = class_preds[indices[1]] + 1
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
end

for i=1,2 do
    print('accuracy ' .. i .. ',' .. 100*class_performance[i]/class_counts[i] .. ' %')
end

print(class_preds)
print(class_performance)
print(class_counts)

-- train accuracy
trainSize = #trainSet.data
numTrainExamples = trainSize[1]
correct = 0
for i=1,numTrainExamples do
    local groundtruth = trainSet.labels[i]
    local prediction = net:forward(trainSet.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print(correct, 100*correct/numTrainExamples .. ' % Training accuracy')

-- train accuracy by class
class_performance = {0, 0}
class_counts = {0, 0}
class_preds = {0,0}
for i=1,trainSet:size() do
    local groundtruth = trainSet.labels[i]
    class_counts[groundtruth] = class_counts[groundtruth] + 1
    local prediction = net:forward(trainSet.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    class_preds[indices[1]] = class_preds[indices[1]] + 1
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
end

for i=1,2 do
    print('accuracy ' .. i .. ',' .. 100*class_performance[i]/class_counts[i] .. ' %')
end
print(class_preds)
print(class_performance)
print(class_counts)
