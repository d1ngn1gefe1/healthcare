import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from processCSV import processData

def mapFeatures(X, theta_u, theta_v, images):
    m = X.shape[0]
    n2 = theta_u.shape[0]
    features = np.zeros((m,n2))
    for i in range(0,m):
        I = cv2.imread(images[i],0)
        width = I.shape[0]
        height = I.shape[1]
        for j in range(0,n2):
            left = X[i,0:2] + theta_u[j]/X[i][2]
            right =  X[i,0:2] + theta_v[j]/X[i][2]
            left[0] = left[0] if left[0] < width else width-1
            left[1] = left[1] if left[1] < height else height-1
            right[0] = right[0] if right[0] < width else width-1
            right[1] = right[1] if right[1] < height else height-1
            features[i][j] = float(I[left[0]][left[1]]) - float(I[right[0]][right[1]])
    return features

n1 = 2 # each pixel has 2D x and y coordinates
n2 = 50 # dimension of feature vector after feature mapping
maxOffset = 100 # max offset for u and v
trainRatio = 0.7

(X,labels,images) = processData()
m = X.shape[0]
# this split doesn't work for very few images
xTrain = X[0:m*trainRatio, :]
xTest = X[m*trainRatio:m, :]
labelsTrain = labels[0:m*trainRatio]
labelsTest = labels[m*trainRatio:m]
imageTrain = images[0:m*trainRatio]
imageTest = images[m*trainRatio:m]

# phi = (theta, tau)
thetaU = np.random.randint(maxOffset, size=(n2,2)) # each row is (u1,u2)
thetaV = np.random.randint(maxOffset, size=(n2,2)) # each row is (v1,v2)
tau = np.random.rand(n2)

# feature mapping
features = mapFeatures(xTrain, thetaU, thetaV, imageTrain)

# train a random forest classifier
rf = RandomForestClassifier(n_estimators=3, criterion='entropy', max_depth=20)
rf.fit(features, labelsTrain)

# prediction
result = rf.predict(mapFeatures(xTest, thetaU, thetaV, imageTest))

# accuracy
accuracy = 0.0
accClass = np.zeros((5,2))
for i in range(0,result.shape[0]):
    accClass[labelsTest[i]][1] += 1
    if result[i] == labelsTest[i]:
        accClass[result[i]][0] += 1
        accuracy += 1
accuracy /= float(result.shape[0])

for i in range(0,5):
    print('Class ' + str(i) + ': ' + str(accClass[i][0]) + '/' + str(accClass[i][1]))
print('Total accuracy: ' + str(accuracy))   

