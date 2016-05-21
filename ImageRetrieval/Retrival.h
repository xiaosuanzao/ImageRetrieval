#ifndef RETRIVAL_H
#define RETRIVAL_H

#include<vector>
#include<string>
#include"Distance.h"
using namespace std;


typedef struct _RetrivalResult
{
	string imgPath;
	string label;
	double distance;
}RetrivalResult;

typedef struct _FeatureData
{
	RetrivalResult data;
	vector<double> feature;
}FeatureData;


class Retrival
{
public:
	Retrival();
	Retrival(const string& featurePath);
	vector<RetrivalResult> retrive(const string& imgPath, Distance& distance = EuclidDistance());
	void showResult(const vector<RetrivalResult>& result);	
	vector<FeatureData> getData(){ return featureData; }

private:
	void load(const string& featurePath);

private:
	vector<FeatureData> featureData;
	int topK;//·µ»ØµÄ¼ìË÷Í¼ÏñÊý
};

#endif