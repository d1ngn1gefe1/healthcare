//
//  helper.hpp
//  UserViewer
//
//  Created by Zelun Luo on 1/7/16.
//  Copyright © 2016 Zelun Luo. All rights reserved.
//

#ifndef helper_hpp
#define helper_hpp

#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml.hpp"
#include "NiTE.h"

using namespace cv;
using namespace std;

#define N_JOINTS 15

void side2top(float side[][5], float top[][5]);
void drawSkeleton(Mat &img, float side[][5]);
void knnsearch(float skeleton[][5], const openni::DepthPixel *imageBuffer, Mat &out, int width, int height);

#endif /* helper_hpp */
