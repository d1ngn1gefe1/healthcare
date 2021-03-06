import numpy as np
import math

def rigidBodyMotion(coord,dx,dy,dz,rx,ry,rz):
    #rotateX = np.array([[1,0,0,0],[0,math.cos(rx),-math.sin(rx),0],[0,math.sin(rx),math.cos(rx),0],[0,0,0,1]])
    #rotateY = np.array([[math.cos(ry),0,math.sin(ry),0],[0,1,0,0],[-math.sin(ry),0,math.cos(ry),0],[0,0,0,1]])
    #rotateZ = np.array([[math.cos(rz),-math.sin(rz),0,0],[math.sin(rz),math.cos(rz),0,0],[0,0,1,0],[0,0,0,1]])
    
    rotation = np.array([[-1,0,0,0],[0,0,-1,0],[0,-1,0,0],[0,0,0,1]])
    #rotation = np.dot(np.dot(rotateZ,rotateY),rotateX)
    #print(rotation)
    
    #t = np.eye(4)
    #t[:,3] = np.array([dx,dy,dz,1])
    t = np.eye(4)
    t[0][3] = dx
    t[1][3] = dy
    t[2][3] = dz
    t[3][3] = -1
    #print(t)
    T = -np.dot(rotation,t)
    #print(T)

    projection = np.dot(rotation,T)
    print(T)

    homoCord = np.ones((4,1))
    for i in range(0,3):
        homoCord[i] = coord[i]
    return np.dot(projection,homoCord)[0:3,:]
    #return np.dot(projection,homoCord)

