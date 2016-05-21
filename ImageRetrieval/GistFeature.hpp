#pragma once

#include <opencv2/core/core.hpp>
#include <vector>
using namespace cv;
using namespace std;
    
class GistFeature 
{
public:

	const char* NAME;
    const char* name()  { return NAME; }

    static vector<double> getFeature(const cv::Mat& image);
    static GistFeature getFeature(const cv::Mat& image, const cv::Mat& mask);

};