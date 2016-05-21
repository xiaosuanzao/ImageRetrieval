#include <string.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "BoxgistFeature.hpp"
#include "GistFeature.hpp"
using namespace cv;
using namespace std;

vector<double> BoxGistFeature::getFeature(const cv::Mat& image, const cv::Mat& mask) {

    cv::Mat image_rgb;
    if (image.channels() < 3) {
        cv::cvtColor(image, image_rgb, CV_GRAY2BGR);
    } else {
        image_rgb = image;
    }

    cv::Mat image_(cv::Size(WIDTH, HEIGHT), CV_8UC3);
    cv::Mat mask_(cv::Size(WIDTH, HEIGHT), CV_8UC1);
    cv::resize(image_rgb, image_, image_.size(), cv::INTER_NEAREST);
    cv::resize(mask, mask_, mask_.size(), cv::INTER_NEAREST);
	
    //Find the maximum square component based on grabcut
    int dp[WIDTH][HEIGHT];
    memset(dp, 0, sizeof(dp));
		
    int maxS = 0;
    for (int i = 1; i < mask_.rows; ++i)
        for (int j = 1; j < mask_.cols; ++j) 
            if (mask_.at<uchar>(i, j) > 200) {
                int val = std::min(std::min(dp[i - 1][j - 1], dp[i - 1][j]), dp[i][j - 1]) + 1;
                dp[i][j] = std::max(dp[i][j], val);
                maxS = std::max(maxS, dp[i][j]);
            }
		//printf("maxS:%d\n",maxS);
    
    int ex = -1, ey = -1;
    for (int i = 0; i < mask_.rows; ++i)
        for (int j = 0; j < mask_.cols; ++j)
            if (dp[i][j] == maxS) {
                ex = i;
                ey = j;
            }
    int sx = ex - maxS + 1, sy = ey - maxS + 1;
        
    int sq_size = std::min((int)LENGTH, maxS);
    cv::Mat square_image(cv::Size(sq_size, sq_size), CV_8UC3);
    //printf("sq size = %d\n", sq_size);
    
    int deltax = (ex - sx + 1 - sq_size) / 2;
    int deltay = (ey - sy + 1 - sq_size) / 2;
    for (int i = 0; i < sq_size; i++)
        for (int j = 0; j < sq_size; j++)
            for (int k = 0; k < 3; k++) {
                square_image.at<cv::Vec3b>(i, j)[k] = image_.at<cv::Vec3b>(i + sx + deltax, j + sy + deltay)[k];
            }
    
	//imshow("input",square_image);
    vector<double> gistfea = GistFeature::getFeature(square_image);

    return gistfea;
}

BoxGistFeature BoxGistFeature::getFeature(const cv::Mat& image) {
	throw("error");
}

