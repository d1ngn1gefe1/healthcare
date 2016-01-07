import numpy as np
import math

def rigidBodyMotion(coord,dx,dy,dz,vx,vy,vz):

    rotation = np.array([vx,vy,vz,[0,0,0,1]])
    
    t = np.zeros((4,4))
    t[0][3] = dx
    t[1][3] = dy
    t[2][3] = dz
    t[3][3] = 0

    translation = -np.dot(rotation,t)
    projection = rotation + translation

    homoCord = np.ones((4,1))
    for i in range(0,3):
        homoCord[i] = coord[i]
    return np.dot(projection,homoCord)[0:3,:]

