import numpy as np
import cameraModel as cm

frontJoints = np.loadtxt('data/joints1.dat',delimiter=',')
num = frontJoints.shape[0]

dx = 0
dy = 1.04e3
dz = 3.52e3

# unit vectors
vx = np.array([-1,0,0,0])
vy = np.array([0,0,-1,0])
vz = np.array([0,-1,0,0])

topJoints = np.zeros((num,3))
for i in range(0,num):
    topJoints[i] = (cm.rigidBodyMotion(frontJoints[i][0:3],dx,dy,dz,vx,vy,vz)).reshape(3)

np.savetxt('data/topJoints2.txt',topJoints,delimiter=',',newline='\n')


