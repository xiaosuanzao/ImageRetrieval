#include<fstream>
#include<algorithm>
#include<iostream>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc.hpp>
#include"Retrival.h"
#include"Utils.h"
#include"Feature.h"

using namespace std;
using namespace cv;

Retrival::Retrival()
{
	topK = 16;
	load("feature.txt");
}

Retrival::Retrival(const string& featurePath)
{
	topK = 16;
	load(featurePath);	
}

void Retrival::load(const string& featurePath)
{

	ifstream in(featurePath.c_str(), ios::in);
	if (in.is_open() == false)
	{
		string msg = "打开文件失败：" + featurePath;
		throw exception(msg.c_str());
	}

	string line;
	while (getline(in, line))
	{
		if (line.empty())
		{
			continue;
		}

		vector<string> part;
		part = split(line, ";");
		vector<string> tmpFeature = split(part[2], ",");
		vector<double> feature(tmpFeature.size());
		for (int i = 0; i < tmpFeature.size(); i++)
		{
			feature[i] = stod(tmpFeature[i].c_str());
		}

		FeatureData data;
		data.data.imgPath = part[0];
		data.data.label = part[1];
		data.data.distance = -1;
		data.feature = feature;

		featureData.push_back(data);
	}

	in.close();
}

bool cmp(RetrivalResult result1, RetrivalResult result2)
{
	return result1.distance < result2.distance;
}

vector<RetrivalResult> Retrival::retrive(const string& imgPath, Distance& distance)
{
	vector<double> feature = Feature::getFeature(imgPath);
	vector<RetrivalResult> result;
	RetrivalResult tmp;

	for (int i = 0; i < featureData.size(); i++)
	{
		tmp.imgPath = featureData[i].data.imgPath;
		tmp.label = featureData[i].data.label;
		tmp.distance = distance.cal(featureData[i].feature, feature);
		result.push_back(tmp);
	}

	sort(result.begin(), result.end(), cmp);
	

	result.erase(result.begin() + topK, result.end());	

	return result;
}


void Retrival::showResult(const vector<RetrivalResult>& result)
{
	int height = 128, width = 128;
	int rows = 4, cols = 4;

	Mat showImg = Mat::zeros(height * rows, width * cols, CV_8UC3);
	//图像拼接
	for (int i = 0; i < result.size(); i++)
	{
		Mat img = imread(result[i].imgPath);
		resize(img, img, Size(height, width));

		Mat roi = Mat(showImg, Rect(i % cols * width, i / cols * height, width, height));
		img.copyTo(roi);
	}
	//绘制分割线
	//水平分割线
	for (int row = 1; row < rows; row++)
	{		
		line(showImg, Point(0, row * height), Point(width * cols, row * height), Scalar(0, 0, 0));
	}
	//垂直分割线
	for (int col = 1; col < cols; col++)
	{
		line(showImg, Point(col * width, 0), Point(col * width, height * rows), Scalar(0, 0, 0));
	}

	imshow("检索结果", showImg);
	waitKey();
}