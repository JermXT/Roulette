
#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cv.h>
// # include "vision/roulette.hpp"

int avgBlueIntensity(cv::Mat image)
{
  
  int b = 0, g = 0, r = 0,pixelCount = 0, pix;
  for(int i =0; i < test;i++){
    
  }
}

int main(int argc, char** argv)
{
  // Get the path
  cv::Mat img = cv::imread(argv[1], 1);
  cv::imshow("Original!", img);
  vision_func(img);

  cv::waitKey(0);
}
