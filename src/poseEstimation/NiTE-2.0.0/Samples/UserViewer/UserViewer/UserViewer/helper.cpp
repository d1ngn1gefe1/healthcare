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

float s = 1;//4.74;

Scalar colorMap[16] = {
    Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255), Scalar(0, 255, 255),
    Scalar(255, 0, 255), Scalar(255, 255, 0), Scalar(0, 128, 128), Scalar(128, 0, 128),
    Scalar(128, 128, 0), Scalar(255, 255, 255), Scalar(128, 128, 128), Scalar(255, 0, 128),
    Scalar(0, 128, 255), Scalar(128, 0, 255), Scalar(128, 255, 0), Scalar(255, 128, 0)
                      };



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
    
    line(img, pt[0], pt[1], colorMap[0], 5);
    line(img, pt[1], pt[2], colorMap[1], 4);
    line(img, pt[1], pt[3], colorMap[2], 4);
    line(img, pt[4], pt[2], colorMap[3], 4);
    line(img, pt[5], pt[3], colorMap[4], 4);
    line(img, pt[4], pt[6], colorMap[5], 4);
    line(img, pt[5], pt[7], colorMap[6], 4);
    
    line(img, pt[8], pt[2], colorMap[7], 3);
    line(img, pt[8], pt[3], colorMap[8], 3);
    line(img, pt[8], pt[9], colorMap[9], 3);
    line(img, pt[8], pt[10], colorMap[10], 3);
    
    line(img, pt[9], pt[10], colorMap[11], 2);
    line(img, pt[11], pt[9], colorMap[12], 2);
    line(img, pt[12], pt[10], colorMap[13], 2);
    line(img, pt[11], pt[13], colorMap[14], 2);
    line(img, pt[12], pt[14], colorMap[15], 2);
}

void knnsearch(float skeleton[][5], const openni::DepthPixel *imageBuffer, Mat &out, int width, int height) {
    Ptr<ml::KNearest> knn(ml::KNearest::create());

    Mat tmp1(N_JOINTS, 5, CV_32FC1, skeleton);
    Mat tmp2, tmp3;
    tmp1(Range(0, tmp1.rows), Range(3, 5)).copyTo(tmp2);
    tmp1(Range(0, tmp1.rows), Range(2, 3)).copyTo(tmp3);
    hconcat(tmp2, tmp3/s, tmp1);
    
    Mat_<float> trainFeatures = (Mat_<float> &)tmp1;
    
    Mat_<int> trainLabels(1, N_JOINTS);
    trainLabels << 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14;
    
    knn->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
    
    //
    
    float img[width*height][3];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            img[j + i*width][0] = i;
            img[j + i*width][1] = j;
            img[j + i*width][2] = ((float)imageBuffer[j + i*width])/s;
        }
    }
    
    Mat testFeatureTmp = Mat(width*height, 3, CV_32FC1, img);
    Mat_<float> testFeature = (Mat_<float> &)testFeatureTmp;
    
    Mat results;
    knn->findNearest(testFeature, 1, results);
    results = results.reshape(0, height);
    out.create(height, width, CV_8UC3);
    
    for (int i = 0; i < results.rows; i++) {
        for (int j = 0; j < results.cols; j++) {
            int idx = (int)results.at<float>(i,j);
            out.at<Vec3b>(i, j).val[0] = colorMap[idx][0];
            out.at<Vec3b>(i, j).val[1] = colorMap[idx][2];
            out.at<Vec3b>(i, j).val[2] = colorMap[idx][3];
        }
    }
}