require 'cutorch'
require 'nn'
require 'image'

-- READ TRAIN IMAGES

imageTypes = {'d'}
dir = '/scail/data/group/vision/u/syyeung/hospital/data/'
datasets = {'cvpr10-17-15afternoon/', 'cvpr10-18-15morning/', 'cvpr10-19-15morning/', 'cvpr10-20-15morning/'}

coor = {115, 240}
boxSz = 64
crop = false

skip = 3
ratio = 0.3 -- pos/neg

labels = {}
filesSet = {}
for k, v in pairs(imageTypes) do
    filesSet[k] = {}
end

for k1, v1 in pairs(datasets) do
    local labelsDir = dir .. v1 .. 'labels.txt'
    file = io.open(labelsDir)
    if file then
        local i = 0
        for line in file:lines() do
            local label = tonumber(line)
            table.insert(labels, label)
            for k2, v2 in pairs(imageTypes) do
                local fileName = dir .. v1 .. v2 .. '/' .. i*skip .. '.jpg'
                table.insert(filesSet[k2], fileName)
                --print(fileName .. ': ' .. label)
            end
            i = i+1
        end
    else
        print('label file does not exist')
    end
end

-- balance pos and neg and randomize the order
labelsTensor = torch.Tensor(#labels)
for i = 1, #labels do
    labelsTensor[i] = labels[i]
end
pos = labelsTensor:nonzero()
neg = torch.add(labelsTensor, -1):nonzero()
print(pos:size(1), neg:size(1))
nPos = pos:size(1)
nNeg = math.floor(nPos/ratio)
print(nPos, nNeg)

labels2 = {}
filesSet2 = {}
for k, v in pairs(imageTypes) do
    filesSet2[k] = {}
end

for i = 1, nPos do
    local idx = pos[i][1]
    assert(labels[idx] == 1)
    table.insert(labels2, labels[idx])
    for k, v in pairs(imageTypes) do
        table.insert(filesSet2[k], filesSet[k][idx])
    end
end

math.randomseed(os.time())

for i = 1, nNeg do
    local rand = math.random(neg:size(1))
    while neg[rand][1] == 0 do
        rand = math.random(neg:size(1))
    end

    local idx = neg[rand][1]
    assert(labels[idx] == 0)
    table.insert(labels2, labels[idx])
    for k, v in pairs(imageTypes) do
        table.insert(filesSet2[k], filesSet[k][idx])
    end
    neg[rand] = 0
end

--labelsTensor = torch.Tensor(#labels2)
--for i = 1, #labels2 do
--    labelsTensor[i] = labels2[i]
--end
--pos = labelsTensor:nonzero()
--neg = torch.add(labelsTensor, -1):nonzero()
--print(pos:size(1), neg:size(1))

labels = labels2
filesSet = filesSet2

for i = 1, #labels*2 do
    local idx1 = math.random(#labels)
    local idx2 = math.random(#labels)
    labels[idx1], labels[idx2] = labels[idx2], labels[idx1] 
    for k, v in pairs(imageTypes) do
        filesSet[k][idx1], filesSet[k][idx2] = filesSet[k][idx2], filesSet[k][idx1]
    end
end

-- read images
imagesSet = {}
for k, v in pairs(imageTypes) do
    imagesSet[k] = {}
end

for k1, v1 in pairs(filesSet) do
    for k2, v2 in pairs(v1) do
        table.insert(imagesSet[k1], image.load(v2))
    end
end

imageSize = imagesSet[1][1]:size()
print(imageSize)
--print(#labels)

nChannelsSet = {}
totalChannels = 0
for k, v in pairs(imageTypes) do
    nChannelsSet[k] = imagesSet[k][1]:size(1)
    totalChannels = totalChannels + nChannelsSet[k]
end
--print(nChannelsSet)

-- create and save data object
hh = {}

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

print('tl: ' .. x1 .. ',' .. y1, 'br: ' .. x2 .. ',' .. y2)

if crop then
    hh.images = torch.Tensor(#labels, totalChannels, boxSz, boxSz)
else
    hh.images = torch.Tensor(#labels, totalChannels, imageSize[2], imageSize[3])
end
for i = 1, #labels do
    if i % 1000 == 0 then
        print('processing train image ' .. i)
    end
    curIdx = 1
    for k, v in pairs(imageTypes) do
        if crop then
            hh.images[i][{{curIdx, curIdx+nChannelsSet[k]-1}, {}, {}}] = imagesSet[k][i][{{}, {y1, y2}, {x1, x2}}]
        else
            hh.images[i][{{curIdx, curIdx+nChannelsSet[k]-1}, {}, {}}] = imagesSet[k][i]
        end
        curIdx = curIdx + nChannelsSet[k]
    end
end

hh.labels = torch.Tensor(#labels)
for i = 1, #labels do
    hh.labels[i] = labels[i]
end

dataFile = '../data/hh'
for i = 1, #imageTypes do
    dataFile = dataFile .. '_' .. imageTypes[i]
end
if crop then
    dataFile = dataFile .. '_crop'
end
dataFile = dataFile .. '.t7'
print('saving file')
torch.save(dataFile, hh)
