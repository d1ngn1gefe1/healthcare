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
float dz = 2.85e3;

float unitVectors[4][4] = {{1,0,0,0}, {0,0,1,0}, {0,-1,0,0}, {0,0,0,1}};

Scalar colorMap[16] = {
    Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255), Scalar(0, 255, 255),
    Scalar(255, 0, 255), Scalar(255, 255, 0), Scalar(0, 128, 128), Scalar(128, 0, 128),
    Scalar(128, 128, 0), Scalar(255, 255, 255), Scalar(128, 128, 128), Scalar(255, 0, 128),
    Scalar(0, 128, 255), Scalar(128, 0, 255), Scalar(128, 255, 0), Scalar(255, 128, 0)
                      };

string jointMap[15] {
    "HEAD", "NECK", "LEFT SHOULDER", "RIGHT SHOULDER",
    "LEFT ELBOW", "RIGHT ELBOW", "LEFT HAND", "RIGHT HAND",
    "TORSO", "LEFT HIP", "RIGHT HIP", "LEFT KNEE",
    "RIGHT KNEE", "LEFT FOOT", "RIGHT FOOT"
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
#define SLOPE 0.2 // important tuning parameter
#define SHIFT 0

enum Normalization {NONE, Z_ONLY, XYZ, SKELETON};

void makeTestFeatures(const openni::DepthPixel *imageBuffer, Mat &mask, Mat &testFeature, vector<Point> &pts, float skeleton[][5], int width, int height, Normalization method) {
    vector<float> testFeatureVtr;

    if (method == NONE) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (mask.at<uchar>(i, j) && imageBuffer[j + i*width]) {
                    pts.push_back(Point(j, i));
                    testFeatureVtr.push_back(j); // x
                    testFeatureVtr.push_back(i); // y
                    testFeatureVtr.push_back((float)imageBuffer[j + i*width]); // z                   
                }
            }
        }
    }
    else if (method == Z_ONLY) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (mask.at<uchar>(i, j) && imageBuffer[j + i*width]) {
                    pts.push_back(Point(j, i));
                    testFeatureVtr.push_back(j);
                    testFeatureVtr.push_back(i);
                    testFeatureVtr.push_back(((float)imageBuffer[j + i*width]+SHIFT)*SLOPE);
                }
            }
        }
    }
    else if (method == XYZ) {
        Mat tmp(width*height, 1, CV_16UC1, (void *)imageBuffer);
        double min, max;
        minMaxLoc(tmp, &min, &max);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (mask.at<uchar>(i, j) && imageBuffer[j + i*width]) {
                    pts.push_back(Point(j, i));
                    testFeatureVtr.push_back((float)j/width);
                    testFeatureVtr.push_back((float)i/height);
                    testFeatureVtr.push_back(((float)imageBuffer[j + i*width] - min)/(max - min));                 
                }
            }
        }
    }
    else if (method == SKELETON) {    
        float actualSkelDepth[N_JOINTS][2] = {0};
      
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (mask.at<uchar>(i, j) && imageBuffer[j + i*width]) {
                    pts.push_back(Point(j, i));
                    testFeatureVtr.push_back(j);
                    testFeatureVtr.push_back(i);
                    testFeatureVtr.push_back((float)imageBuffer[j + i*width]);
                    
                    for (int k = 0; k < N_JOINTS; k++) {
                        if (j < skeleton[k][3] + 5 && j > skeleton[k][3] - 5 &&
                            i < skeleton[k][4] + 5 && i > skeleton[k][4] - 5) {
                            actualSkelDepth[k][0] += (float)imageBuffer[j + i*width];
                            actualSkelDepth[k][1] += 1;
                        }
                    }
                }
            }
        }

        for (int k = 0; k < N_JOINTS; k++) {
            if (actualSkelDepth[k][1] != 0) {
                actualSkelDepth[k][0] /= actualSkelDepth[k][1];
            };
        }

        // normalize according to "HEAD" (0), "LEFT SHOULDER" (2), and "RIGHT SHOULDER" (3)
        //float shift = (skeleton[0][2] - actualSkelDepth[0][0]) +
        //              (skeleton[2][2] - actualSkelDepth[3][0]) +
        //              (skeleton[2][2] - actualSkelDepth[3][0]);
        //shift /= 3;
        float shift = (skeleton[0][2] - actualSkelDepth[0][0]);
        //printf("shift: %lf\n", shift);
        for (vector<float>::iterator it = testFeatureVtr.begin() + 2; it < testFeatureVtr.end(); it += 3) {
            *it += shift;
            *it *= SLOPE;
        }
    }

    testFeature = Mat(testFeatureVtr.size()/3, 3, CV_32FC1, (void *)&testFeatureVtr[0]);
}

void makeTrainFeatures(float skeleton[][5],  Mat &trainFeatures, Normalization method) {
    Mat skel(N_JOINTS, 5, CV_32FC1, (void *)skeleton);
    Mat skel1, skel2;
    skel(Range(0, skel.rows), Range(3, 5)).copyTo(skel1);
    skel(Range(0, skel.rows), Range(2, 3)).copyTo(skel2);

    if (method == NONE) {
        hconcat(skel1, skel2, trainFeatures); // x, y, z
    }
    else if (method == Z_ONLY || method == SKELETON) {
        hconcat(skel1, skel2*SLOPE, trainFeatures);
    }
    else if(method == XYZ) {
        double min, max;
        minMaxLoc(skel1.col(0), &min, &max);
        skel1.col(0) = (skel1.col(0) - min)/(max - min);
        
        minMaxLoc(skel1.col(1), &min, &max);
        skel1.col(1) = (skel1.col(1) - min)/(max - min);
        
        minMaxLoc(skel2.col(0), &min, &max);
        assert(min >= 0);
        skel2.col(0) = (skel2.col(0) - min)/(max - min);
        
        hconcat(skel1, skel2, trainFeatures);
    }
}

void knnsearch(float skeleton[][5], const openni::DepthPixel *imageBuffer, Mat &mask, Mat &out, int *label, int width, int height) {
    // skeleton: x real world, y real world, z real world, x depth, y depth
    
    Ptr<ml::KNearest> knn(ml::KNearest::create());
    Normalization method = SKELETON;
    memset(label, -1, width*height*sizeof(int));
    
    // make train labels
    Mat_<int> trainLabels(1, N_JOINTS);
    trainLabels << 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14;

    // make train features 
    Mat trainFeaturesTmp;
    makeTrainFeatures(skeleton, trainFeaturesTmp, method);
    Mat_<float> trainFeatures = (Mat_<float> &)trainFeaturesTmp;

    // make test features   
    Mat testFeaturesTmp;
    vector<Point> pts;
    makeTestFeatures(imageBuffer, mask, testFeaturesTmp, pts, skeleton, width, height, method);
    Mat_<float> testFeatures = (Mat_<float> &)testFeaturesTmp;
    
    // train knn
    knn->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
    
    // check
    //getInfo(trainFeaturesTmp);
    //getInfo(testFeaturesTmp);
    //printf("\n\n");
    
    // get results
    Mat results;
    knn->findNearest(testFeatures, 1, results);
    out = Mat::zeros(height, width + TEXT_W, CV_8UC3);
    
    // visualize the result by mapping label to color
    for (int i = 0; i < pts.size(); i++) {
        int x = pts[i].x;
        int y = pts[i].y;
        int z = results.at<float>(i, 0);
        out.at<Vec3b>(y, x).val[0] = colorMap[z][0];
        out.at<Vec3b>(y, x).val[1] = colorMap[z][1];
        out.at<Vec3b>(y, x).val[2] = colorMap[z][2];
        label[y*width + x] = (int)z;
    }
    
    // draw text
    for (int i = 0; i < N_JOINTS; i++) {
        putText(out, jointMap[i], Point(width, (float)height*(i+1)/N_JOINTS-5), FONT_HERSHEY_SIMPLEX, 0.3, colorMap[i]);
    }
}
