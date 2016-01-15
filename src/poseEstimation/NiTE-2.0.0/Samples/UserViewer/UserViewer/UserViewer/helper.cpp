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
float dz = 2.8e3;

float unitVectors[4][4] = {{1,0,0,0}, {0,0,1,0}, {0,-1,0,0}, {0,0,0,1}};

float s = 4.74;

Scalar colorMap[16] = {
    Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255), Scalar(0, 255, 255),
    Scalar(255, 0, 255), Scalar(255, 255, 0), Scalar(0, 128, 128), Scalar(128, 0, 128),
    Scalar(128, 128, 0), Scalar(255, 255, 255), Scalar(128, 128, 128), Scalar(255, 0, 128),
    Scalar(0, 128, 255), Scalar(128, 0, 255), Scalar(128, 255, 0), Scalar(255, 128, 0)
                      };

string jointMap[15] {
    "HEAD",
    "NECK",
    "LEFT SHOULDER",
    "RIGHT SHOULDER",
    "LEFT ELBOW",
    "RIGHT ELBOW",
    "LEFT HAND",
    "RIGHT HAND",
    "TORSO",
    "LEFT HIP",
    "RIGHT HIP",
    "LEFT KNEE",
    "RIGHT KNEE",
    "LEFT FOOT",
    "RIGHT FOOT"
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

void getInfo(Mat &img) {
    Scalar mean;
    Scalar stddev;
    double min;
    double max;
    
    meanStdDev(img.col(0), mean, stddev);
    minMaxLoc(img.col(0), &min, &max);
    printf("x: %d, %d, %lf, %lf\n", (int)(mean.val[0]), (int)(stddev.val[0]), min, max);
    meanStdDev(img.col(1), mean, stddev);
    minMaxLoc(img.col(1), &min, &max);
    printf("y: %d, %d, %lf, %lf\n", (int)(mean.val[0]), (int)(stddev.val[0]), min, max);
    meanStdDev(img.col(2), mean, stddev);
    minMaxLoc(img.col(2), &min, &max);
    printf("z: %d, %d, %lf, %lf\n", (int)(mean.val[0]), (int)(stddev.val[0]), min, max);
}

#define ACTUAL_SKEL 0
#define TEXT_W 100

void drawText(Mat &img, int width, int height) {
    for (int i = 0; i < N_JOINTS; i++) {
        putText(img, jointMap[i], Point(width, (float)height*(i+1)/N_JOINTS-5), FONT_HERSHEY_SIMPLEX, 0.3, colorMap[i]);
    }
}

void knnsearch(float skeleton[][5], const openni::DepthPixel *imageBuffer, Mat &mask, Mat &out, int width, int height) {
    Ptr<ml::KNearest> knn(ml::KNearest::create());

    // make train features and train labels
    Mat skel(N_JOINTS, 5, CV_32FC1, skeleton);
    Mat skel1, skel2;
    skel(Range(0, skel.rows), Range(3, 5)).copyTo(skel1);
    skel(Range(0, skel.rows), Range(2, 3)).copyTo(skel2);
    hconcat(skel1, skel2, skel);
    
    Mat_<float> trainFeatures = (Mat_<float> &)skel;
    
    Mat_<int> trainLabels(1, N_JOINTS);
    trainLabels << 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14;
    
    // train knn
    knn->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
    
    // make test features
    int n = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (mask.at<uchar>(i, j) != 0 && imageBuffer[j + i*width] != 0) {
                n++;
            }
        }
    }
    
#if ACTUAL_SKEL
    float actualSkelFl[N_JOINTS][2] = {0};
#endif
    
    int idx = 0;
    float testFeatureFl[n][3];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (mask.at<uchar>(i, j) == 0 || imageBuffer[j + i*width] == 0) {
                continue;
            }
            
#if ACTUAL_SKEL
            for (int k = 0; k < N_JOINTS; k++) {
                if (j < (int)skel.at<float>(k, 0)+5 && j > (int)skel.at<float>(k, 0)-5 &&
                    i < (int)skel.at<float>(k, 1)+5 && i > (int)skel.at<float>(k, 1)-5) {
                    actualSkelFl[k][0] += (float)imageBuffer[j + i*width];
                    actualSkelFl[k][1] += 1;
                }
            }
#endif
            
            testFeatureFl[idx][0] = j;
            testFeatureFl[idx][1] = i;
            testFeatureFl[idx][2] = (float)imageBuffer[j + i*width];
            idx++;
        }
    }
    
#if ACTUAL_SKEL
    for (int k = 0; k < N_JOINTS; k++) {
        if (actualSkelFl[k][1] != 0) {
            actualSkelFl[k][0] /= actualSkelFl[k][1];
        }
        printf("joint %d: %f vs %f\n", k, actualSkelFl[k][0], skeleton[k][2]);
    }
#endif
    
    assert(n == idx);
    
    Mat testFeatureTmp = Mat(n, 3, CV_32FC1, testFeatureFl);
    Mat_<float> testFeature = (Mat_<float> &)testFeatureTmp;
    
    getInfo(skel);
    getInfo(testFeatureTmp);
    printf("\n\n");
    
    Mat results;
    knn->findNearest(testFeature, 1, results);
    out = Mat::zeros(height, width + TEXT_W, CV_8UC3);
    
    // visualize the result
    for (int i = 0; i < n; i++) {
        int x = (int)testFeatureFl[i][0];
        int y = (int)testFeatureFl[i][1];
        int z = results.at<float>(i, 0);
        out.at<Vec3b>(y, x).val[0] = colorMap[z][0];
        out.at<Vec3b>(y, x).val[1] = colorMap[z][1];
        out.at<Vec3b>(y, x).val[2] = colorMap[z][2];
    }
    
    drawText(out, width, height);
}
