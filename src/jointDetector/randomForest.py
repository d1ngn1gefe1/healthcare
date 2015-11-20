from sklearn.ensemble import RandomForestClassifier
import numpy as np

def mapFeatures(X, theta_u, theta_v, I):
    width = I.shape[0]
    height = I.shape[1]
    m = X.shape[0]
    n2 = theta_u.shape[0]
    features = np.zeros((m,n2))
    for i in range(0,m):
        for j in range(0,n2):
            left = X[i] + theta_u[j]/I[X[i][0]][X[i][1]]
            right =  X[i] + theta_v[j]/I[X[i][0]][X[i][1]]
            left[0] = left[0] if left[0] < width else width-1
            left[1] = left[1] if left[1] < height else height-1
            right[0] = right[0] if right[0] < width else width-1
            right[1] = right[1] if right[1] < height else height-1
            f = I[left[0]][left[1]] - I[right[0]][right[1]]
            features[i][j] = f 
    return features

m = 2000 # number of training examples (i.e. number of pixels)
n1 = 2 # each pixel has 2D x and y coordinates
n2 = 50 # dimension of feature vector after feature mapping
width = 320 # width of each image
height = 240
num_class = 5 

scale = np.array([width, height])
X = np.random.rand(m,n1) * np.transpose(scale) # each row is a pixel
I = np.random.rand(width,height) # a corresponding image of depth values

max_offset = 100
theta_u = np.random.randint(max_offset, size=(n2,2)) # each row is (u1,u2)
theta_v = np.random.randint(max_offset, size=(n2,2)) # each row is (v1,v2)
tau = np.random.rand(n2)

features = mapFeatures(X, theta_u, theta_v, I)
y = np.random.randint(num_class, size=m) # labels for each pixel

rf = RandomForestClassifier(n_estimators=3, criterion='entropy', max_depth=20)
rf.fit(features, y)

