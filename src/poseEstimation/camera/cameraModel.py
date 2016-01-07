import numpy as np
import math

root2 = math.sqrt(2)
d1 = 2e3
d2 = 2.7e3

rotation = np.array([[-1,0,0],[0,0,-1],[0,-1,0]])
t = np.array([[0],[d1],[d2]])

translation = -np.dot(rotation,t)
projection = np.zeros((4,4))

for i in range(0,3):
    for j in range(0,3):
        projection[i][j] = rotation[i][j]

for i in range(0,3):
    projection[i][3] = translation[i]

projection[3][3] = 1

# e.g. coord0 = np.random.rand(3,1)
def rigidBodyMotion(coord):
    homoCord = np.ones((4,1))
    for i in range(0,3):
        homoCord[i] = coord[i]
    return np.dot(projection,homoCord)[0:3,:]
    #return np.dot(projection,homoCord)

