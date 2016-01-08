//
//  helper.cpp
//  UserViewer
//
//  Created by Zelun Luo on 1/7/16.
//  Copyright © 2016 Zelun Luo. All rights reserved.
//

#include "helper.hpp"

float dx = 0;
float dy = 0.4e3;
float dz = 2.9e3;

float unitVectors[4][4] = {{-1,0,0,0}, {0,0,-1,0}, {0,-1,0,0}, {0,0,0,1}};

void rigidBodyMotion(float sideJoint[4], float topJoint[4], float dx, float dy, float dz, float unitVectors[4][4]) {
    float rotation[4][4];
    float projection[4][4];
    float t[4] = {dx, dy, dz, 0};
    float translation[4];
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            rotation[i][j] = unitVectors[i][j];
            projection[i][j] = rotation[i][j];
        }
    }
    
    for (int i = 0; i < 4; i++) {
        float sum = 0;
        for (int j = 0; j < 4; j++) {
            sum += -rotation[i][j] * t[j];
        }
        translation[i] = sum;
        projection[i][3] = translation[i];
    }
    
    for (int i = 0; i < 4; i++) {
        float sum = 0;
        for (int j = 0; j < 4; j++) {
            sum += projection[i][j] * sideJoint[j];
        }
        topJoint[i] = sum;
    }
}

void side2top(float side[][5], float top[][5]) {
    for (int i = 0; i < N_JOINTS; i++) {
        float sideJoint[4] = {side[i][0], side[i][1], side[i][2], 1};
        float topJoint[4];
        rigidBodyMotion(sideJoint, topJoint, dx, dy, dz, unitVectors);
        
        for (int j = 0; j < 3; j++) {
            top[i][j] = topJoint[j];
        }
    }
}