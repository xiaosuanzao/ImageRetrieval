#pragma once

#include "GistFeature.hpp"
    
class BoxGistFeature {
public:
    
    static const char* NAME;
    const char* name() const { return NAME; }

    static BoxGistFeature getFeature(const cv::Mat& image);
    static vector<double> getFeature(const cv::Mat& image, const cv::Mat& mask);

    int lx, ly, rx, ry, clx, cly, crx, cry;
private:
    enum{WIDTH = 220, HEIGHT = 220, LENGTH = 100};
};