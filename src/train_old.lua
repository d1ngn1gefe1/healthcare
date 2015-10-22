require 'cutorch'
require 'nn'
require 'cunn'

-- load data file
dataFile = '/scail/scratch/group/vision/hospital/data/hh_d.t7'
hh = torch.load(dataFile)

trainset = {}
train_pos = hh.train.labels:nonzero()
train_neg = torch.add(hh.train.labels,-1):nonzero()

-- create training set with every other example a negative
neg_stride = math.floor(train_neg:size(1) / train_pos:size(1))
numTrainPairs = train_pos:size(1)
numTrainExamples = numTrainPairs*2
trainData = torch.Tensor(numTrainExamples,1,240,320)
trainLabels = torch.Tensor(numTrainExamples)
for i = 1,numTrainPairs do
    if i % 10 == 0 then
        print('processing training pair ' .. i)
    end

    local pos_idx = train_pos[i]
    local neg_idx = train_neg[(i-1)*neg_stride+1]

    trainData[2*i-1] = hh.train.data:narrow(1,pos_idx[1],1):float()
    trainLabels[2*i-1] = hh.train.labels:narrow(1,pos_idx[1],1) + 1

    trainData[2*i] = hh.train.data:narrow(1,neg_idx[1],1):float()
    trainLabels[2*i] = hh.train.labels:narrow(1,neg_idx[1],1) + 1
end

trainset.data = trainData
trainset.labels = trainLabels

print(trainset)
    
testset = hh.test

setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.labels[i]} 
                end}
);
trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size() 
    return self.data:size(1) 
end

setmetatable(testset, 
    {__index = function(t, i) 
                    return {t.data[i], t.labels[i]} 
                end}
);
testset.data = testset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function testset:size() 
    return self.data:size(1) 
end

mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,1 do -- over each image channel
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- view an image
-- itorch.image(trainset.data[301])

-- define network
net = nn.Sequential()
net:add(nn.SpatialConvolution(1, 6, 7, 7, 2, 2))
net:add(nn.SpatialMaxPooling(3,3,3,3))  
net:add(nn.SpatialConvolution(6, 16, 5, 5, 2, 2))
net:add(nn.SpatialMaxPooling(2,2,2,2))
-- net:add(nn.SpatialConvolution(16, 16, 5, 5))
-- net:add(nn.SpatialMaxPooling(2,2,2,2))
net = net:cuda()
-- out = net:forward(trainset.data[1]) -- must cuda the data first
-- print(out:size())
    
net:add(nn.View(16*9*12))                   
net:add(nn.Linear(16*9*12, 50))            
net:add(nn.Linear(50, 2)) 
--net:add(nn.Linear(120, 84))
--net:add(nn.Linear(84, 2))                  
net:add(nn.LogSoftMax())                    

criterion = nn.ClassNLLCriterion()

--train
net = net:cuda()
criterion = criterion:cuda()
trainset.data = trainset.data:cuda()

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 10

trainer:train(trainset)

-- test accuracy

testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
for i=1,1 do -- over each image channel
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end
testset.data = testset.data:cuda()

testSize = #testset.data
numTestExamples = testSize[1]
correct = 0
for i=1,numTestExamples do
    local groundtruth = testset.labels[i] + 1
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print(correct, 100*correct/numTestExamples .. ' % ')

-- test accuracy by class

class_preds = {0, 0}
class_performance = {0, 0}
class_counts = {0, 0}
for i=1,numTestExamples do
    local groundtruth = testset.labels[i] + 1
    class_counts[groundtruth] = class_counts[groundtruth] + 1
    local prediction = net:forward(testset.data[i])
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

trainSize = #trainset.data
numTrainExamples = trainSize[1]
correct = 0
for i=1,numTrainExamples do
    local groundtruth = trainset.labels[i]
    local prediction = net:forward(trainset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print(correct, 100*correct/numTrainExamples .. ' % ')


-- train accuracy by class

class_performance = {0, 0}
class_counts = {0, 0}
class_preds = {0,0}
for i=1,trainset:size() do
    local groundtruth = trainset.labels[i]
    class_counts[groundtruth] = class_counts[groundtruth] + 1
    local prediction = net:forward(trainset.data[i])
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
