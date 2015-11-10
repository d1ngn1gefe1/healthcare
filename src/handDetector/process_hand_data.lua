require 'image'

imageType = 'negative'
imageFrameDir = '/scail/scratch/group/vision/hospital/hands/'

images = {}
labels = {}
i = 1
for imageName in io.popen('ls ' .. imageFrameDir .. imageType):lines() do
  if string.match(imageName, '.jpg') then
    fileName = imageFrameDir .. imageType .. '/' .. imageName
    print(fileName)
    images[i] = image.load(fileName)
    if imageType == 'positive' then
      labels[i] = 1
    else
      labels[i] = 0
    end
    i = i + 1
  end 
end

imageSize = images[1]:size()
hh = {}
hh.data = torch.Tensor(#images, imageSize[1], imageSize[2], imageSize[3])
hh.labels = torch.Tensor(#images)
for i = 1, #images do
  hh.data[i] = images[i]
  hh.labels[i] = labels[i]
end

dataFile = '../data/hh_' .. imageType .. '.t7'
torch.save(dataFile, hh)
