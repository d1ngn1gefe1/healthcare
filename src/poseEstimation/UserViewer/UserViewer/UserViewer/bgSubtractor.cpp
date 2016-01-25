//
//  bgSubtractor.cpp
//  UserViewer
//
//  Created by Zelun Luo on 1/13/16.
//  Copyright © 2016 Zelun Luo. All rights reserved.
//

#include "bgSubtractor.hpp"

BgSubtractor::BgSubtractor(int w, int h) {
    // create Background Subtractor objects
    pMOG2 = createBackgroundSubtractorMOG2(); //MOG2 approach
    width = w;
    height = h;
}

BgSubtractor::~BgSubtractor() {
}

void BgSubtractor::processImages(Mat &img) {
    pMOG2->apply(img, fgMaskMOG2, 0.5);
}

void BgSubtractor::getBg(Mat &bg) {
    pMOG2->getBackgroundImage(bg);
    //bg = fgMaskMOG2;
}

void removeSmallBlobs(Mat& im, double size)
{
    // Only accept CV_8UC1
    if (im.channels() != 1 || im.type() != CV_8U)
        return;
    
    // Find all contours
    vector<vector<Point> > contours;
    findContours(im.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    
    //printf("%d: ", contours.size());
    
    //Mat imcolor;
    //cvtColor(im, imcolor, CV_GRAY2RGB);
    
    for (int i = 0; i < contours.size(); i++)
    {
        // Calculate contour area
        double area = contourArea(contours[i]);
        
        // Remove small objects by drawing the contour with black color
        if (area >= 0 && area <= size)
            drawContours(im, contours, i, 0, -1);
    }
}

void BgSubtractor::getMask(Mat &img, Mat &mask) {
    Mat bg;
    pMOG2->getBackgroundImage(bg);
    //GaussianBlur(bg, bg, Size(5, 5), 10);
    //GaussianBlur(img, img, Size(5, 5), 10);
    absdiff(img, bg, mask);
    threshold(mask, mask, 1, 255, THRESH_BINARY);
    equalizeHist(mask, mask);
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    erode(mask, mask, element);
    removeSmallBlobs(mask, 1000);
}
