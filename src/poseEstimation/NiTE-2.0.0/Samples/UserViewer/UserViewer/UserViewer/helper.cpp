//
//  helper.cpp
//  UserViewer
//
//  Created by Zelun Luo on 1/7/16.
//  Copyright © 2016 Zelun Luo. All rights reserved.
//

#include "helper.hpp"

float dx = 0;
float dy = 1.5e3;
float dz = 2.6e3;

float unitVectors[4][4] = {{1,0,0,0}, {0,0,1,0}, {0,-1,0,0}, {0,0,0,1}};

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

void drawSkeleton(Mat &img, float side[][5]) {
    Point pt[N_JOINTS];
    for (int i = 0; i < N_JOINTS; i++) {
        pt[i] = Point(side[i][3], side[i][4]);
        circle(img, pt[i], 5, Scalar(255, 255, 255), -1);
    }
    
    line(img, pt[0], pt[1], Scalar(0, 255, 0), 5);
    line(img, pt[1], pt[2], Scalar(255, 0, 0), 4);
    line(img, pt[1], pt[3], Scalar(255, 0, 0), 4);
    line(img, pt[4], pt[2], Scalar(255, 0, 0), 4);
    line(img, pt[5], pt[3], Scalar(255, 0, 0), 4);
    line(img, pt[4], pt[6], Scalar(255, 0, 0), 4);
    line(img, pt[5], pt[7], Scalar(255, 0, 0), 4);
    
    line(img, pt[8], pt[2], Scalar(255, 255, 0), 3);
    line(img, pt[8], pt[3], Scalar(255, 255, 0), 3);
    line(img, pt[8], pt[9], Scalar(255, 255, 0), 3);
    line(img, pt[8], pt[10], Scalar(255, 255, 0), 3);
    
    line(img, pt[9], pt[10], Scalar(0, 0, 255), 2);
    line(img, pt[11], pt[9], Scalar(0, 0, 255), 2);
    line(img, pt[12], pt[10], Scalar(0, 0, 255), 2);
    line(img, pt[11], pt[13], Scalar(0, 0, 255), 2);
    line(img, pt[12], pt[14], Scalar(0, 0, 255), 2);
}