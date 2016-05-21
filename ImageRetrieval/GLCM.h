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
	vector<double> getFeature(int angle,int distance,int graylevel);//���angele����distance���룬graylevel���Ҷȼ�������

private:
	Mat getGLCM(int angle, int distance, int graylevel);//��ù�������angle��ȡ0,45,90,135��distanceȡ������graylevel����ѹ���ĻҶȵȼ���������ʾ��ѹ��

private:
	Mat img;
	Mat grayCompress(int level);//��256����imgѹ��Ϊlevel���ȼ�
};

#endif