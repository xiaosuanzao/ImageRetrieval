#ifndef GLCM_H
#define GLCM_H

#include <opencv2/core/core.hpp>
#include<string>
#include<vector>
using namespace cv;
using namespace std;

class GLCM
{
public:
	GLCM(const string& imgpath);
	GLCM(const Mat src);	
	vector<double> getFeature(int angle,int distance,int graylevel);//获得angele方向，distance距离，graylevel个灰度级的特征

private:
	Mat getGLCM(int angle, int distance, int graylevel);//获得共生矩阵。angle可取0,45,90,135。distance取正数。graylevel设置压缩的灰度等级。负数表示不压缩

private:
	Mat img;
	Mat grayCompress(int level);//将256级的img压缩为level个等级
};

#endif