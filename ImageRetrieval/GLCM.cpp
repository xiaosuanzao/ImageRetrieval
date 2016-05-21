#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "GLCM.h"
using namespace cv;

GLCM::GLCM(const string& imgpath)
{
	img = imread(imgpath,0);
	if(img.data == NULL)
		img = NULL;
}

GLCM::GLCM(const Mat src)
{
	if(src.data != NULL)
	{
		if (src.channels() == 3)
			cvtColor(src,img,CV_BGR2GRAY);
		else if(src.channels() == 1)
			src.copyTo(img);
	}		
	else
		img = NULL;
}


//��imgѹ��Ϊgraylevel���Ҷȼ�,�Ҷȷ�Χ[0,graylevel)
Mat GLCM::grayCompress(int graylevel)
{
	if(graylevel < 1 || graylevel > 255)//����Խ�緵��ԭʼ�Ҷ�ͼ��
		return img;
	double k = (graylevel - 1)/ 255.0;
	Mat compressimg(img.size(),CV_8UC1);
	for(int row = 0;row < img.rows;row++)
		for(int col = 0;col < img.cols;col++)
		{
			compressimg.at<uchar>(row,col) = (uchar)(img.at<uchar>(row,col) * k + 0.5);
		}
	return compressimg;
}

Mat GLCM::getGLCM(int angle,int distance,int graylevel)
{
	int dx = 0, dy = 0;//�����ʼ�����ƫ����
	switch(angle)
	{
		case 0:dx = distance;dy = 0;break;
		case 45:dx = distance;dy = distance;break;
		case 90:dx = 0;dy = distance;break;
		case 135:dx = -distance;dy = distance;break;
		default:dx = distance;dy = 0;
	}

	if(graylevel < 1 || graylevel > 256)
		graylevel = 256;//ͼ��ѹ��
	Mat grayimg = grayCompress(graylevel);
	Mat glcm;//�Ҷȹ�������
	glcm = Mat::zeros(Size(graylevel,graylevel),CV_64FC1);//��ʼ���Ҷȹ�������
	for (int row = 0; row < grayimg.rows; row++)
	{
		for (int col = 0; col < grayimg.cols; col++)
		{
			double centergray = grayimg.at<uchar>(row, col);//��ʼ��ĻҶ�
			double neighborgray = 0.0;//��ʼ���ھӵĻҶ�
			//����(row,col)���꣬angle�Ƕȣ�distance����ĵ������
			int neighborrow = row + dx;
			int neighborcol = col + dy;
			if (neighborrow >= 0 && neighborrow < grayimg.rows && neighborcol >= 0 && neighborcol < grayimg.cols)//�ھӵ�λ��ͼ���ڲ�
				neighborgray = grayimg.at<uchar>(neighborrow, neighborcol);
			glcm.at<double>(centergray, neighborgray)++;
		}
	}		

	int totalpixle = 0;
	if(angle == 0)
		totalpixle = grayimg.rows * (grayimg.cols - distance);
	else if(angle == 90)
		totalpixle = (grayimg.rows - distance) * grayimg.cols;
	else
		totalpixle = (grayimg.rows - distance) * (grayimg.cols - distance);
	for (int row = 0; row < graylevel; row++)
	{
		for (int col = 0; col < graylevel; col++)
		{
			glcm.at<double>(row, col) /= totalpixle;
		}
	}
			
	return glcm;
}

vector<double> GLCM::getFeature(int angle,int distance,int graylevel)
{
	Mat glcm = getGLCM(angle,distance,graylevel);
	vector<double> feature;

	double entropy = 0.0;
	//double IDM = 0.0;
	double ASM = 0.0;
	double CON = 0.0;
	//double COR = 0.0;
	double HOM = 0.0;//ͬ����
	//double mu_x = 0.0,mu_y = 0.0;
	//double sigma_x = 0.0,sigma_y = 0.0;
	const double EPS = 1E-8;
	for(int row = 0;row < glcm.rows;row++)
		for(int col = 0;col < glcm.cols;col++)
		{
			double value = glcm.at<double>(row,col);			
			if(value > EPS)//��value = 0�����
				entropy -= value * log(value);
			//IDM += value / (1 + pow(row - col,2.0));
			ASM += pow(value,2);
			CON += value * pow(row - col,2.0);
			//mu_x += row * value;
			//mu_y += col * value;
			//COR += row * col * value;
			HOM += value / (1 + abs(row - col));
		}

	/*for(int row = 0;row < glcm.rows;row++)
		for(int col = 0;col < glcm.cols;col++)
		{
			double value = glcm.at<double>(row,col);
			sigma_x += pow(row - mu_x,2) * value;
			sigma_y += pow(col - mu_y,2) * value;
		}

	COR = (COR - mu_x * mu_y) / sigma_x / sigma_y;*/

	feature.push_back(entropy);

	feature.push_back(ASM);
	feature.push_back(CON);

	feature.push_back(HOM);

	return feature;
}
