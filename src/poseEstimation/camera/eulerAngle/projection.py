import numpy as np
import cameraModel as cm
import math

dx = 0
dy = 2e3
dz = 3e3

#Euler angles??
alpha = math.radians(0)
gamma = math.radians(0)
beta = math.radians(0)

frontJoints = np.loadtxt('joints.dat',delimiter=',')
num = frontJoints.shape[0]

topJoints = np.zeros((num,3))
for i in range(0,num):
    topJoints[i] = (cm.rigidBodyMotion(frontJoints[i],dx,dy,dz,alpha,beta,gamma)).reshape(3)

np.savetxt('topJoints.txt',topJoints,delimiter=',',newline='\n')


