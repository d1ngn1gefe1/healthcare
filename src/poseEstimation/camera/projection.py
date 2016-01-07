import numpy as np
import cameraModel as cm

frontJoints = np.loadtxt('joints.dat',delimiter=',')
num = frontJoints.shape[0]

topJoints = np.zeros((num,3))
for i in range(0,num):
    topJoints[i] = (cm.rigidBodyMotion(frontJoints[i])).reshape(3)

np.savetxt('topJoints.txt',topJoints,delimiter=',',newline='\n')


