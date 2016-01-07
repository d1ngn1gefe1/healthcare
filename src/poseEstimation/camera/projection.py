import numpy as np
import cameraModel as cm

dx = 0
dy = 2e3
dz = 2.7e3
rx = math.radians(180)
ry = math.radians(45)
rz = math.radians(235)

frontJoints = np.loadtxt('joints.dat',delimiter=',')
num = frontJoints.shape[0]

topJoints = np.zeros((num,3))
for i in range(0,num):
    topJoints[i] = (cm.rigidBodyMotion(frontJoints[i],dx,dy,dz,rx,ry,rz)).reshape(3)

np.savetxt('topJoints.txt',topJoints,delimiter=',',newline='\n')


